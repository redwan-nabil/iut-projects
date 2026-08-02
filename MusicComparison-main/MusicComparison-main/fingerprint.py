"""
Step 3b: Core fingerprinting functions.

Path A: Spectrogram peak-hashing (Shazam-style) -> for exact audio clips
Path B: Pitch contour extraction -> for humming/singing (matched via DTW later)

These functions are shared between database-building (Step 3) and
live-query processing (Step 4) - that's intentional, the query must be
processed identically to the database entries.
"""

import numpy as np
import librosa
from scipy.ndimage import maximum_filter


# ---------------------------------------------------------------------------
# PATH A: Spectrogram peak-hashing
# ---------------------------------------------------------------------------

def compute_spectrogram(y, sr, n_fft=2048, hop_length=512):
    """Return magnitude spectrogram in dB."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db


def find_peaks(S_db, amp_min=-40, neighborhood_size=20):
    """
    Find local maxima in the spectrogram (time-frequency peaks).
    Returns list of (freq_bin, time_frame) tuples.
    """
    local_max = maximum_filter(S_db, size=neighborhood_size) == S_db
    detected = (S_db > amp_min) & local_max
    freq_idx, time_idx = np.where(detected)
    peaks = list(zip(freq_idx, time_idx))
    return peaks


def generate_hashes(peaks, fan_value=10, min_time_delta=0, max_time_delta=200):
    """
    Pair each anchor peak with several nearby target peaks and hash them.
    Hash = combination of (freq1, freq2, time_delta) -> robust, compact fingerprint.

    Returns list of (hash_str, anchor_time_frame).
    """
    peaks = sorted(peaks, key=lambda p: p[1])  # sort by time
    hashes = []

    for i in range(len(peaks)):
        for j in range(1, fan_value + 1):
            if i + j >= len(peaks):
                break
            freq1, t1 = peaks[i]
            freq2, t2 = peaks[i + j]
            dt = t2 - t1
            if min_time_delta <= dt <= max_time_delta:
                h = f"{freq1}|{freq2}|{dt}"
                hashes.append((h, t1))

    return hashes


def fingerprint_hashes(y, sr):
    """Full Path A pipeline: audio -> list of (hash, anchor_time)."""
    S_db = compute_spectrogram(y, sr)
    peaks = find_peaks(S_db)
    return generate_hashes(peaks)


# ---------------------------------------------------------------------------
# PATH B: Pitch contour extraction (for humming/singing, matched via DTW)
# ---------------------------------------------------------------------------

def extract_pitch_contour(y, sr, fmin=65, fmax=1000, frame_length=2048, hop_length=512):
    """
    Extract a pitch (melody) contour using librosa's pYIN algorithm.
    Returns an array of relative pitch changes in semitones (key-invariant),
    with unvoiced/silent frames removed.

    NOTE: CREPE (via the `crepe` pip package) gives higher accuracy than pYIN
    if you have time to add a torch/tensorflow dependency - swap this function
    out for CREPE later if your evaluation shows pYIN isn't accurate enough.
    """
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length
    )

    # Keep only voiced frames (i.e. where a pitch was actually detected)
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) < 2:
        return np.array([])

    # Convert Hz -> MIDI note number (log scale), then take frame-to-frame deltas
    midi = librosa.hz_to_midi(voiced_f0)
    contour = np.diff(midi)  # relative pitch changes in semitones = key-invariant

    return contour


# ---------------------------------------------------------------------------
# Recommendation embeddings: MFCC + chroma summary vector
# ---------------------------------------------------------------------------

def extract_embedding(y, sr, n_mfcc=13):
    """
    Compact timbre + harmony summary vector for a song, used only for the
    "similar songs" recommendation (cosine similarity), not for identification.

    mean+std of MFCCs (timbre) concatenated with mean+std of chroma (harmony),
    L2-normalized so a plain dot product gives cosine similarity.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    vec = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        chroma.mean(axis=1), chroma.std(axis=1),
    ])

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ---------------------------------------------------------------------------
# Shared audio loading
# ---------------------------------------------------------------------------

def load_audio(filepath, sr=22050, mono=True):
    """Load and resample audio to a consistent format."""
    y, sr = librosa.load(filepath, sr=sr, mono=mono)
    return y, sr
