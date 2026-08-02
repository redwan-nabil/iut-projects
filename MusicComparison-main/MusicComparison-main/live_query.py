"""
Step 4 + 5: Live query capture, fingerprinting, and matching (CLI demo).

Walks you through TWO recordings in one run, each with an interactive
prompt beforehand, and prints the identified song for each:
  1. Clip recording  -> Path A: spectrogram peak hashes -> hash-vote match
  2. Humming/singing -> Path B: pitch contour -> DTW match

This is Zones 2+3 in your system diagram: "Microphone Interface / GUI"
-> "Pre-processing & Fingerprinting" -> "Matching Engine". For a GUI
version of the same pipeline, see app.py (streamlit run app.py).

Run:
    python live_query.py

Output:
    query_clip.wav          (raw clip recording)
    query_hum.wav           (raw humming recording)
    query_fingerprint.pkl   (dict with both hashes + contour, for inspection)
"""

import pickle

import sounddevice as sd
import soundfile as sf

from fingerprint import load_audio, fingerprint_hashes, extract_pitch_contour
from match_engine import MatchEngine

SAMPLE_RATE = 22050  # must match fingerprint.py's load_audio() default
CHANNELS = 1
RECORD_SECONDS = 8.0   # change this if you want longer/shorter recordings
DEVICE = None          # change this to a device index if your mic isn't the default


def record(seconds: float, out_path: str, device: int = None):
    print(f"Recording for {seconds} seconds...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=device,
    )
    sd.wait()
    sf.write(out_path, audio, SAMPLE_RATE)
    print(f"Saved recording to {out_path}")


def prompt_and_record(label: str, instructions: str, seconds: float, out_path: str, device: int = None):
    """Show a prompt, wait for Enter, then record."""
    print(f"\n--- {label} ---")
    print(instructions)
    input("Press Enter when you're ready to start recording...")
    record(seconds, out_path, device)


def fingerprint_clip(wav_path: str):
    print("Computing spectrogram peak hashes (Path A)...")
    y, sr = load_audio(wav_path)
    hashes = fingerprint_hashes(y, sr)
    print(f"  -> {len(hashes)} hashes generated")
    return hashes


def fingerprint_hum(wav_path: str):
    print("Extracting pitch contour (Path B)...")
    y, sr = load_audio(wav_path)
    contour = extract_pitch_contour(y, sr)
    print(f"  -> contour length: {len(contour)} frames")
    return contour


def print_matches(title, matches, score_key):
    print(f"\n--- {title} ---")
    if not matches:
        print("  No matches found.")
        return
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m['title']} / {m['artist']}  ({score_key}={m[score_key]})")


if __name__ == "__main__":
    print("Loading matching engine (song database)...")
    engine = MatchEngine()

    # --- Step 1: clip recording ---
    prompt_and_record(
        label="Step 1/2: Exact clip",
        instructions="Play a short clip of the song near the microphone (not humming).",
        seconds=RECORD_SECONDS,
        out_path="query_clip.wav",
        device=DEVICE,
    )
    clip_hashes = fingerprint_clip("query_clip.wav")
    clip_result = engine.identify("clip", query_hashes=clip_hashes)
    print_matches("Clip identification results", clip_result["matches"], "score")
    if clip_result["recommendations"]:
        print_matches("Similar songs", clip_result["recommendations"], "similarity")

    # --- Step 2: humming/singing recording ---
    prompt_and_record(
        label="Step 2/2: Humming / singing",
        instructions="Hum or sing the melody of the same song.",
        seconds=RECORD_SECONDS,
        out_path="query_hum.wav",
        device=DEVICE,
    )
    hum_contour = fingerprint_hum("query_hum.wav")
    hum_result = engine.identify("hum", query_contour=hum_contour)
    print_matches("Humming identification results", hum_result["matches"], "distance")
    if hum_result["recommendations"]:
        print_matches("Similar songs", hum_result["recommendations"], "similarity")

    # --- Save both fingerprints together (for inspection/debugging) ---
    result = {
        "clip_hashes": clip_hashes,
        "hum_contour": hum_contour,
    }
    with open("query_fingerprint.pkl", "wb") as f:
        pickle.dump(result, f)

    print("\nSaved query_fingerprint.pkl (clip_hashes + hum_contour).")
