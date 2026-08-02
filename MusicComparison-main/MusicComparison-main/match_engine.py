"""
Step 5: Matching engine.

Loads the three offline databases (hashes, pitch contours, embeddings) once,
then exposes:
  - match_clip(query_hashes)    -> Path A: hash-vote matching for exact clips
  - match_hum(query_contour)    -> Path B: DTW matching for humming/singing
  - recommend(song_id)          -> cosine similarity over embeddings

Hash matching (Shazam-style):
  For every (hash, anchor_time) in the query, look up which (song_id, db_time)
  pairs share that hash. If the query is really a clip of that song, the
  offsets (db_time - anchor_time) will cluster around one constant value for
  the correct song (constant relative alignment) and be scattered/random for
  wrong songs. So for each song we take the single most common offset and use
  its count as the match score.

DTW matching:
  The query contour (relative pitch deltas, semitones) is compared against
  every song's full contour using librosa's subsequence DTW (open-begin,
  open-end): one DP pass finds the best-matching *subsequence* of the song
  for the whole query, rather than requiring the query to align with the
  whole song. Lower normalized distance = better match.

  librosa's DTW core is numba-jitted, so the very first call in a process
  pays a one-off JIT compilation cost (~15-20s) - MatchEngine warms this up
  once at construction so real queries aren't hit with that latency.

  Contours can optionally be decimated (every Nth frame kept) before DTW to
  shrink the O(N*M) cost matrix. Measured on this database: decimation=1
  (no decimation) gives perfect top-1 discrimination on clean melody slices
  at ~4s/query against all 51 songs; decimation=2 drops to ~58% top-1 for a
  ~2x speedup, decimation=4 collapses to ~8% top-1 (frame-to-frame pitch
  deltas alias badly once you skip frames - a missed note-onset silently
  becomes a different melody). So DTW_DECIMATION defaults to 1: at this
  database size, 4s/query is an acceptable "near real-time" latency and
  correctness matters more than shaving seconds. This speed/accuracy
  tradeoff (and the raw numbers) is exactly what evaluate.py measures.
"""

import pickle
from collections import Counter, defaultdict

import numpy as np
import csv
import librosa

HASH_DB_PATH = "hash_database.pkl"
CONTOUR_DB_PATH = "contour_database.pkl"
EMBEDDINGS_DB_PATH = "embeddings_database.pkl"
METADATA_CSV = "metadata.csv"

DTW_DECIMATION = 1


class MatchEngine:
    def __init__(self):
        with open(HASH_DB_PATH, "rb") as f:
            self.hash_db = pickle.load(f)
        with open(CONTOUR_DB_PATH, "rb") as f:
            self.contour_db = pickle.load(f)
        with open(EMBEDDINGS_DB_PATH, "rb") as f:
            self.embeddings_db = pickle.load(f)

        self._warm_up_dtw()

        self.metadata = {}
        with open(METADATA_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.metadata[row["song_id"]] = row

    def _song_info(self, song_id):
        row = self.metadata.get(song_id, {})
        return {
            "song_id": song_id,
            "title": row.get("title", "Unknown"),
            "artist": row.get("artist", "Unknown"),
            "filepath": row.get("filepath", ""),
        }

    # -----------------------------------------------------------------
    # Path A: hash-vote matching (exact clip)
    # -----------------------------------------------------------------

    def match_clip(self, query_hashes, top_k=5):
        """
        query_hashes: list of (hash_str, anchor_time) from fingerprint_hashes().
        Returns list of dicts {song_id, title, artist, score, offset}, best first.
        """
        # song_id -> Counter(offset -> vote count)
        offset_votes = defaultdict(Counter)

        for h, anchor_time in query_hashes:
            matches = self.hash_db.get(h)
            if not matches:
                continue
            for song_id, db_time in matches:
                offset = db_time - anchor_time
                offset_votes[song_id][offset] += 1

        results = []
        for song_id, votes in offset_votes.items():
            best_offset, score = votes.most_common(1)[0]
            info = self._song_info(song_id)
            info["score"] = score
            info["offset"] = best_offset
            results.append(info)

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    # -----------------------------------------------------------------
    # Path B: DTW contour matching (humming/singing)
    # -----------------------------------------------------------------

    @staticmethod
    def _warm_up_dtw():
        """Trigger librosa's numba JIT compilation once, off the query path."""
        dummy = np.zeros((1, 8), dtype=np.float64)
        librosa.sequence.dtw(X=dummy, Y=dummy, subseq=True, metric="euclidean")

    @staticmethod
    def _subsequence_distance(query, reference, decimation=DTW_DECIMATION):
        """
        Best-matching-subsequence DTW distance of `query` against `reference`,
        normalized by (decimated) query length so scores are comparable
        across queries of different lengths. Lower = better match.
        """
        query = np.asarray(query, dtype=np.float64)[::decimation]
        reference = np.asarray(reference, dtype=np.float64)[::decimation]

        if len(query) < 2 or len(reference) < 2:
            return float("inf")
        if len(reference) < len(query):
            # subseq DTW requires the reference to be at least as long as the query
            query, reference = reference, query

        D, _ = librosa.sequence.dtw(
            X=query.reshape(1, -1), Y=reference.reshape(1, -1),
            subseq=True, metric="euclidean",
        )
        return float(D[-1, :].min()) / len(query)

    def match_hum(self, query_contour, top_k=5):
        """
        query_contour: 1D array of semitone deltas from extract_pitch_contour().
        Returns list of dicts {song_id, title, artist, distance}, best (lowest
        distance) first.
        """
        results = []
        for song_id, contour in self.contour_db.items():
            if contour is None or len(contour) == 0:
                continue
            distance = self._subsequence_distance(query_contour, contour)
            info = self._song_info(song_id)
            info["distance"] = distance
            results.append(info)

        results.sort(key=lambda r: r["distance"])
        return results[:top_k]

    # -----------------------------------------------------------------
    # Recommendations: cosine similarity over embeddings
    # -----------------------------------------------------------------

    def recommend(self, song_id, top_k=4):
        query_vec = self.embeddings_db.get(song_id)
        if query_vec is None:
            return []

        scored = []
        for other_id, vec in self.embeddings_db.items():
            if other_id == song_id:
                continue
            similarity = float(np.dot(query_vec, vec))  # vectors are L2-normalized
            info = self._song_info(other_id)
            info["similarity"] = similarity
            scored.append(info)

        scored.sort(key=lambda r: r["similarity"], reverse=True)
        return scored[:top_k]

    # -----------------------------------------------------------------
    # Convenience: full identify pipeline for one query
    # -----------------------------------------------------------------

    def identify(self, mode, query_hashes=None, query_contour=None,
                 top_k=5, recommend_k=4):
        """
        mode: "clip" or "hum".
        Returns dict {matches: [...], recommendations: [...]} where
        recommendations are based on the top match (empty if no match found).
        """
        if mode == "clip":
            matches = self.match_clip(query_hashes, top_k=top_k)
        elif mode == "hum":
            matches = self.match_hum(query_contour, top_k=top_k)
        else:
            raise ValueError(f"Unknown mode: {mode!r} (expected 'clip' or 'hum')")

        recommendations = []
        if matches:
            recommendations = self.recommend(matches[0]["song_id"], top_k=recommend_k)

        return {"matches": matches, "recommendations": recommendations}
