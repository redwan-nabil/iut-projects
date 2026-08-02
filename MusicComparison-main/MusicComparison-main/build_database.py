"""
Step 3c: Build the full offline database.

For every song in metadata.csv, this computes:
  1. Hash fingerprints (Path A - exact clip matching)
  2. Pitch contour (Path B - humming/singing matching)

Saves everything to disk as pickle files so Step 5 (matching) can load
them instantly without recomputing.

Run (after build_metadata.py):
    python build_database.py
Output:
    hash_database.pkl   -> dict: hash -> list of (song_id, anchor_time)
    contour_database.pkl -> dict: song_id -> pitch contour array
"""

import csv
import pickle
import time
from collections import defaultdict

from tqdm import tqdm

from fingerprint import load_audio, fingerprint_hashes, extract_pitch_contour

METADATA_CSV = "metadata.csv"
HASH_DB_OUT = "hash_database.pkl"
CONTOUR_DB_OUT = "contour_database.pkl"


def main():
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        songs = list(csv.DictReader(f))

    if not songs:
        raise SystemExit("metadata.csv is empty. Run build_metadata.py first.")

    hash_database = defaultdict(list)   # hash -> [(song_id, anchor_time), ...]
    contour_database = {}               # song_id -> pitch contour array

    failed = []

    for song in tqdm(songs, desc="Building database"):
        song_id = song["song_id"]
        filepath = song["filepath"]

        try:
            y, sr = load_audio(filepath)

            # --- Path A: hashing ---
            hashes = fingerprint_hashes(y, sr)
            for h, anchor_time in hashes:
                hash_database[h].append((song_id, anchor_time))

            # --- Path B: pitch contour ---
            contour = extract_pitch_contour(y, sr)
            contour_database[song_id] = contour

        except Exception as e:
            print(f"\n[FAILED] {song['title']} ({song_id}): {e}")
            failed.append(song_id)

    with open(HASH_DB_OUT, "wb") as f:
        pickle.dump(dict(hash_database), f)

    with open(CONTOUR_DB_OUT, "wb") as f:
        pickle.dump(contour_database, f)

    print(f"\nDone. {len(songs) - len(failed)}/{len(songs)} songs processed successfully.")
    print(f"  {HASH_DB_OUT}: {len(hash_database)} unique hashes")
    print(f"  {CONTOUR_DB_OUT}: {len(contour_database)} song contours")
    if failed:
        print(f"  Failed song_ids: {failed}")


if __name__ == "__main__":
    main()
