"""
Step 7: Experimental investigation and performance evaluation.

Two evaluations, mirroring the two matching paths:

1. Clip matching (Path A, hash-voting): for every song, extract a random
   10s clip and test it clean and at three additive-white-noise SNR levels.
   Measures Top-1 / Top-5 accuracy and end-to-end latency vs. noise.

2. Humming matching (Path B, DTW): a live smoke test showed pYIN pitch
   tracking on a *polyphonic* (raw song) segment does not reliably recover
   the sung melody line - it picks up whatever's loudest, often
   instrumentation. That's a real limitation of monophonic pitch trackers
   worth reporting, but it means we can't use raw song audio as a stand-in
   for a human hum. Instead we use the melody line already extracted for
   the database (contour_database.pkl) as ground truth, take a random slice
   of it, and apply distortions that approximate how a human would
   reproduce it imperfectly:
     - tempo warp (resampling the slice, +/-15%)   -> humans don't hum at
       the exact original tempo
     - pitch jitter (gaussian noise added per frame) -> humans don't hit
       pitches exactly
   This isolates and measures the DTW matching algorithm's robustness to
   those specific distortions. It does NOT measure end-to-end mic-to-pYIN
   accuracy on real human humming - that must be spot-checked live in the
   demo (see README note printed at the end of this script).

Outputs:
    clip_eval_results.csv, hum_eval_results.csv  (raw per-trial results)
    accuracy_vs_noise.png, accuracy_vs_distortion.png, latency_comparison.png
"""

import csv
import random
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fingerprint import load_audio, fingerprint_hashes
from match_engine import MatchEngine

random.seed(42)
np.random.seed(42)

CLIP_SECONDS = 10.0
CLIP_SNR_CONDITIONS = [("clean", None), ("snr_20db", 20), ("snr_10db", 10), ("snr_0db", 0)]

HUM_SLICE_FRAMES = 300
MIN_SLICE_STD = 1.0  # semitones; reject near-flat/sustained-note slices as unrepresentative of a hummed melody
HUM_DISTORTION_CONDITIONS = [
    ("exact", 0.0, 0.0),        # (label, pitch_jitter_sigma_semitones, tempo_warp_range_half_width)
    ("mild", 0.3, 0.10),
    ("heavy", 0.8, 0.20),
]


def add_noise(clip, snr_db):
    if snr_db is None:
        return clip
    signal_power = np.mean(clip ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=clip.shape).astype(clip.dtype)
    return clip + noise


def distort_contour(contour, jitter_sigma, warp_half_width):
    warp_factor = 1.0 + np.random.uniform(-warp_half_width, warp_half_width)
    new_len = max(2, int(len(contour) * warp_factor))
    warped = np.interp(
        np.linspace(0, len(contour) - 1, new_len),
        np.arange(len(contour)),
        contour,
    )
    if jitter_sigma > 0:
        warped = warped + np.random.normal(0, jitter_sigma, size=warped.shape)
    return warped


def evaluate_clip_matching(engine, songs):
    rows = []
    for song in songs:
        song_id = song["song_id"]
        y, sr = load_audio(song["filepath"])
        dur = len(y) / sr
        if dur <= CLIP_SECONDS + 5:
            continue
        start = random.uniform(5, dur - CLIP_SECONDS - 5)
        clean_clip = y[int(start * sr):int((start + CLIP_SECONDS) * sr)]

        for condition_label, snr_db in CLIP_SNR_CONDITIONS:
            noisy_clip = add_noise(clean_clip, snr_db)

            t0 = time.time()
            hashes = fingerprint_hashes(noisy_clip, sr)
            matches = engine.match_clip(hashes, top_k=5)
            latency = time.time() - t0

            top1 = bool(matches) and matches[0]["song_id"] == song_id
            top5 = any(m["song_id"] == song_id for m in matches)

            rows.append({
                "song_id": song_id, "title": song["title"], "condition": condition_label,
                "n_hashes": len(hashes), "top1_correct": top1, "top5_correct": top5,
                "latency_sec": latency,
            })
            print(f"[clip] {song['title'][:40]:40s} {condition_label:10s} "
                  f"top1={top1} top5={top5} latency={latency:.2f}s")
    return rows


def evaluate_hum_matching(engine, songs):
    rows = []
    for song in songs:
        song_id = song["song_id"]
        full_contour = engine.contour_db.get(song_id)
        if full_contour is None or len(full_contour) <= HUM_SLICE_FRAMES + 10:
            continue

        # Pick a slice with enough pitch movement to be a plausible "hummable"
        # excerpt - a near-flat/sustained-note slice is inherently ambiguous
        # against other songs' held notes and isn't representative of a hum.
        base_slice = None
        for _ in range(10):
            start = random.randint(0, len(full_contour) - HUM_SLICE_FRAMES)
            candidate = full_contour[start:start + HUM_SLICE_FRAMES]
            if np.std(candidate) >= MIN_SLICE_STD:
                base_slice = candidate
                break
        if base_slice is None:
            continue

        for condition_label, jitter_sigma, warp_half_width in HUM_DISTORTION_CONDITIONS:
            query = distort_contour(base_slice, jitter_sigma, warp_half_width)

            t0 = time.time()
            matches = engine.match_hum(query, top_k=5)
            latency = time.time() - t0

            top1 = bool(matches) and matches[0]["song_id"] == song_id
            top5 = any(m["song_id"] == song_id for m in matches)

            rows.append({
                "song_id": song_id, "title": song["title"], "condition": condition_label,
                "query_len": len(query), "top1_correct": top1, "top5_correct": top5,
                "latency_sec": latency,
            })
            print(f"[hum]  {song['title'][:40]:40s} {condition_label:10s} "
                  f"top1={top1} top5={top5} latency={latency:.2f}s")
    return rows


def summarize_and_plot(clip_rows, hum_rows):
    clip_df = pd.DataFrame(clip_rows)
    hum_df = pd.DataFrame(hum_rows)
    clip_df.to_csv("clip_eval_results.csv", index=False)
    hum_df.to_csv("hum_eval_results.csv", index=False)

    clip_summary = clip_df.groupby("condition").agg(
        top1_accuracy=("top1_correct", "mean"),
        top5_accuracy=("top5_correct", "mean"),
        mean_latency_sec=("latency_sec", "mean"),
        n_trials=("song_id", "count"),
    ).reindex([c for c, _ in CLIP_SNR_CONDITIONS])

    hum_summary = hum_df.groupby("condition").agg(
        top1_accuracy=("top1_correct", "mean"),
        top5_accuracy=("top5_correct", "mean"),
        mean_latency_sec=("latency_sec", "mean"),
        n_trials=("song_id", "count"),
    ).reindex([c for c, _, _ in HUM_DISTORTION_CONDITIONS])

    print("\n=== Clip matching (Path A: hash-voting) ===")
    print(clip_summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n=== Humming matching (Path B: DTW, synthetic-distortion proxy) ===")
    print(hum_summary.to_string(float_format=lambda x: f"{x:.3f}"))

    clip_summary.to_csv("clip_eval_summary.csv")
    hum_summary.to_csv("hum_eval_summary.csv")

    # --- Chart 1: accuracy vs noise level ---
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(clip_summary))
    ax.plot(x, clip_summary["top1_accuracy"], marker="o", label="Top-1 accuracy")
    ax.plot(x, clip_summary["top5_accuracy"], marker="o", label="Top-5 accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(clip_summary.index)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Noise condition")
    ax.set_title("Clip identification accuracy vs. noise (hash-voting)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("accuracy_vs_noise.png", dpi=150)
    plt.close(fig)

    # --- Chart 2: accuracy vs distortion level (humming) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(hum_summary))
    ax.plot(x, hum_summary["top1_accuracy"], marker="o", label="Top-1 accuracy")
    ax.plot(x, hum_summary["top5_accuracy"], marker="o", label="Top-5 accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(hum_summary.index)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Distortion condition (tempo warp + pitch jitter)")
    ax.set_title("Humming identification accuracy vs. distortion (DTW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("accuracy_vs_distortion.png", dpi=150)
    plt.close(fig)

    # --- Chart 3: latency comparison ---
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Clip match\n(hash-voting)", "Hum match\n(DTW)"]
    values = [clip_df["latency_sec"].mean(), hum_df["latency_sec"].mean()]
    ax.bar(labels, values, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Mean latency (seconds)")
    ax.set_title("End-to-end matching latency")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}s", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig("latency_comparison.png", dpi=150)
    plt.close(fig)

    print("\nSaved: clip_eval_results.csv, hum_eval_results.csv, "
          "clip_eval_summary.csv, hum_eval_summary.csv, "
          "accuracy_vs_noise.png, accuracy_vs_distortion.png, latency_comparison.png")


def main():
    with open("metadata.csv", newline="", encoding="utf-8") as f:
        songs = list(csv.DictReader(f))

    engine = MatchEngine()

    clip_rows = evaluate_clip_matching(engine, songs)
    hum_rows = evaluate_hum_matching(engine, songs)

    summarize_and_plot(clip_rows, hum_rows)

    print("\nNOTE: humming accuracy above measures DTW robustness to tempo/pitch "
          "distortion applied to the *ground-truth* melody contour, not full "
          "mic -> pYIN -> DTW accuracy on real human humming (pYIN on raw "
          "polyphonic clips does not reliably isolate the sung melody - see "
          "module docstring). Validate real humming performance live via "
          "live_query.py / app.py during the demo and report it qualitatively "
          "alongside these numbers.")


if __name__ == "__main__":
    main()
