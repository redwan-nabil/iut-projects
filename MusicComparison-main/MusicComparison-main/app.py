"""
Step 6: GUI (Zone 2 + Zone 3 of the system diagram).

Streamlit app:
  - Record button -> "Listening..." -> capture mic audio
  - Choose mode: exact clip vs humming/singing
  - Runs the same fingerprinting used offline, then the matching engine
  - Displays identified song (title/artist) + similar-song recommendations

Run:
    streamlit run app.py
"""

import time

import numpy as np
import sounddevice as sd
import streamlit as st

from fingerprint import fingerprint_hashes, extract_pitch_contour
from match_engine import MatchEngine

SAMPLE_RATE = 22050
CLIP_SECONDS = 8.0
HUM_SECONDS = 12.0


@st.cache_resource(show_spinner="Loading song database and matching engine...")
def get_engine():
    return MatchEngine()


def record_audio(seconds: float) -> np.ndarray:
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                    channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


st.set_page_config(page_title="Song Identifier", page_icon="🎵")
st.title("🎵 Real-Time Song Identifier")
st.caption("Identify a song from a short clip or from humming/singing the melody.")

engine = get_engine()

mode_label = st.radio(
    "What are you providing?",
    ["🎧 Play a clip of the song", "🎤 Hum or sing the melody"],
)
mode = "clip" if mode_label.startswith("🎧") else "hum"
seconds = CLIP_SECONDS if mode == "clip" else HUM_SECONDS

st.write(f"Recording length: {seconds:.0f} seconds.")

if st.button("● Record", type="primary"):
    progress_placeholder = st.empty()
    progress_placeholder.info("🔴 Listening...")
    audio = record_audio(seconds)
    progress_placeholder.info("⚙️ Processing...")

    t_start = time.time()
    if mode == "clip":
        hashes = fingerprint_hashes(audio, SAMPLE_RATE)
        result = engine.identify("clip", query_hashes=hashes)
        empty_check = len(hashes) == 0
    else:
        contour = extract_pitch_contour(audio, SAMPLE_RATE)
        result = engine.identify("hum", query_contour=contour)
        empty_check = len(contour) < 2
    latency = time.time() - t_start

    progress_placeholder.empty()

    if empty_check or not result["matches"]:
        st.error("Couldn't extract enough signal from that recording. "
                  "Try again closer to the mic, or with less background noise.")
    else:
        best = result["matches"][0]
        st.success(f"Identified in {latency:.2f}s")

        st.subheader(best["title"])
        st.write(f"**Artist:** {best['artist']}")

        with st.expander("Other candidates"):
            for m in result["matches"][1:]:
                score_key = "score" if mode == "clip" else "distance"
                st.write(f"- {m['title']} / {m['artist']} ({score_key}={m[score_key]})")

        if result["recommendations"]:
            st.subheader("Similar songs")
            for r in result["recommendations"]:
                st.write(f"- {r['title']} / {r['artist']}  (similarity={r['similarity']:.3f})")

st.divider()
st.caption(f"Database: {len(engine.contour_db)} songs indexed.")
