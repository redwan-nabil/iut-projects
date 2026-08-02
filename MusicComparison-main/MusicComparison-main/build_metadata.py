"""
Step 3a: Build metadata table from the downloaded WAV files.

Your yt-dlp files are named like:
  "Aurthohin - Epitaph (Official Music Video) │ Shopnochura │.cWAxUOjj8aQ.wav"
  "Mon Shudhu Mon Chuyeche - Souls [HQ].jx1j5yLlBPg.wav"

We extract:
  - song_id  -> the YouTube video ID (already unique, embedded in filename)
  - title    -> cleaned-up title (best-effort split on " - ")
  - artist   -> best-effort guess (may need manual correction after)
  - filepath -> full path to the wav file

Run:
    python build_metadata.py
Output:
    metadata.csv
"""

import os
import re
import csv

DATABASE_DIR = "database"       # folder containing your .wav files
OUTPUT_CSV = "metadata.csv"


def parse_filename(filename: str):
    """
    Filenames are: '<title stuff>.<video_id>.wav'
    The video_id is always the second-to-last dot-separated segment.
    """
    name_no_ext = filename.rsplit(".wav", 1)[0]
    parts = name_no_ext.rsplit(".", 1)

    if len(parts) == 2:
        raw_title, video_id = parts
    else:
        raw_title, video_id = name_no_ext, filename  # fallback

    # Clean junk characters often left by YouTube titles
    clean_title = raw_title.replace("｜", "|").strip()
    clean_title = re.sub(r"\(.*?(official|video|hq|hd).*?\)", "", clean_title, flags=re.I)
    clean_title = re.sub(r"\[.*?(hq|hd).*?\]", "", clean_title, flags=re.I)
    clean_title = clean_title.strip(" -|")

    # Best-effort artist/title split on common separators
    artist, title = "Unknown", clean_title

    by_match = re.search(r"\bby\s+([A-Za-z][\w]*(?:\s[A-Za-z][\w]*){0,2})", clean_title, flags=re.I)
    if by_match:
        artist = by_match.group(1).strip()
        title = clean_title[:by_match.start()].strip(" -|:")
        return video_id, title, artist

    for sep in [" - ", " | "]:
        if sep in clean_title:
            left, right = clean_title.split(sep, 1)
            artist, title = left.strip(), right.strip()
            return video_id, title, artist

    # No-space hyphen, e.g. "Warfaze-Obak Bhalobasha": only split if the left
    # side looks like a bare name (no spaces) rather than part of a title.
    hyphen_match = re.match(r"^([A-Za-z][\w.]*)-(.+)$", clean_title)
    if hyphen_match:
        artist, title = hyphen_match.group(1).strip(), hyphen_match.group(2).strip()
        return video_id, title, artist

    return video_id, title, artist


def main():
    if not os.path.isdir(DATABASE_DIR):
        raise SystemExit(f"Folder '{DATABASE_DIR}' not found. Run this from the "
                          f"directory containing your 'database/' folder.")

    rows = []
    for fname in sorted(os.listdir(DATABASE_DIR)):
        if not fname.lower().endswith(".wav"):
            continue
        video_id, title, artist = parse_filename(fname)
        rows.append({
            "song_id": video_id,
            "title": title,
            "artist": artist,
            "filepath": os.path.join(DATABASE_DIR, fname),
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["song_id", "title", "artist", "filepath"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} songs to {OUTPUT_CSV}")
    print("!! Open metadata.csv and manually fix any wrong artist/title splits before continuing !!")


if __name__ == "__main__":
    main()
