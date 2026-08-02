"""
Step 3d: Build the recommendation embedding database.

Separate from build_database.py (hashes + contours) so it can be re-run on
its own without redoing the expensive hash fingerprinting.

For every song in metadata.csv, computes an MFCC+chroma summary vector
(see fingerprint.extract_embedding) used for "similar songs" recommendations
via cosine similarity.

Run (after build_metadata.py):
    python build_embeddings.py
Output:
    embeddings_database.pkl -> dict: song_id -> np.ndarray vector
"""

import csv
import pickle

from tqdm import tqdm

from fingerprint import load_audio, extract_embedding

METADATA_CSV = "metadata.csv"
EMBEDDINGS_OUT = "embeddings_database.pkl"


def main():
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        songs = list(csv.DictReader(f))

    if not songs:
        raise SystemExit("metadata.csv is empty. Run build_metadata.py first.")

    embeddings = {}
    failed = []

    for song in tqdm(songs, desc="Building embeddings"):
        song_id = song["song_id"]
        filepath = song["filepath"]
        try:
            y, sr = load_audio(filepath)
            embeddings[song_id] = extract_embedding(y, sr)
        except Exception as e:
            print(f"\n[FAILED] {song['title']} ({song_id}): {e}")
            failed.append(song_id)

    with open(EMBEDDINGS_OUT, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"\nDone. {len(songs) - len(failed)}/{len(songs)} songs embedded.")
    print(f"  {EMBEDDINGS_OUT}: {len(embeddings)} vectors")
    if failed:
        print(f"  Failed song_ids: {failed}")


if __name__ == "__main__":
    main()
