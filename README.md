# IUT Projects Repository

This repository is a multi-project academic portfolio containing CAD design work, a MATLAB engineering app, a Python audio-recognition system, and a Proteus digital-logic game project.

## 1) Repository Overview

Top-level folders:

- `AutoCAD Project IUT/` — AutoCAD design files (`.dwg`) and backups (`.bak`)
- `Matlab GUI project/` — MATLAB App Designer project (`.mlapp`) and project report (`.docx`)
- `MusicComparison-main/` — Python-based music identification and recommendation project
- `Snake-Ladder game/` — Proteus design/simulation project files (`.pdsprj`)

---

## 2) Folder and File Details

## `AutoCAD Project IUT/`

Engineering drawing assets for EEE/Civil-style design tasks.

### Main files
- `3500-sqft-floor-plan.dwg` (+ `.bak`) — Floor-plan drawing and backup
- `Rain-Water-Harvesting-Tank (1).dwg` (+ `.bak`) — Water tank design
- `Rooftop.dwg` (+ `.bak`) — Rooftop layout
- `Surge-Arrester.dwg` (+ `.bak`) — Surge arrester drawing
- `framework.dwg` (+ `.bak`) — Structural/framework design

### Subfolder
- `4418 Final files/` — Final/alternate versions and variants, including:
  - `2007 files`
  - `ALL MAIN.dwg`, `ALL MAIN (1).dwg`
  - `ALL EMERGENCY(2).dwg` variants
  - `ACAD-ALL EMERGENCY(2).dwg`
  - wiring-related files such as `4418project1_wire done.dwg`

**Purpose:** Stores drafting deliverables and revision history.

---

## `Matlab GUI project/`

Contains a complete MATLAB App Designer implementation of an RLC analyzer.

### Files
- `RLC_Analyzer.mlapp` — MATLAB GUI app source package
- `Project Documentation.docx` — Report/user documentation

### What the app does
From the app source and documentation:
- Takes user inputs: Resistance `R`, Inductance `L`, Capacitance `C`, Peak Voltage `V`, Frequency `f`
- Computes:
  - Angular frequency `ω = 2πf`
  - Reactances `X_L` and `X_C`
  - Total impedance `Z`
  - Peak current `I_peak`
  - Phase angle `φ` (degrees)
- Plots voltage and current waveforms over time

**Purpose:** Interactive circuit analysis and waveform visualization.

---

## `MusicComparison-main/MusicComparison-main/`

Python audio-pattern recognition project (clip matching + humming matching + recommendation).

### Core files
- `app.py` — Streamlit GUI
- `live_query.py` — CLI-based live query demo
- `fingerprint.py` — Signal processing utilities (fingerprints, contours, embeddings)
- `match_engine.py` — Matching logic and recommendation
- `build_metadata.py` — Build metadata CSV
- `build_database.py` — Build fingerprint/contour databases
- `build_embeddings.py` — Build embedding database
- `evaluate.py` — Evaluation and benchmarking

### Data and outputs
- `metadata.csv`
- `clip_eval_results.csv`, `clip_eval_summary.csv`
- `hum_eval_results.csv`, `hum_eval_summary.csv`
- `accuracy_vs_noise.png`, `accuracy_vs_distortion.png`, `latency_comparison.png`

### Other
- `requirements.txt` — Python dependencies
- `README.md` — Detailed project-specific documentation
- `LICENSE` — MIT license for this subproject

**Purpose:** Local music identification and similarity recommendation.

---

## `Snake-Ladder game/Snake-Ladder game/`

Proteus project for Snake & Ladder implementation across multiple phases.

### Main subfolders
- `Phase A & B/`
  - Core phase project files (`.pdsprj`)
  - Workspace configuration XML files
  - Backup snapshots (`.pdsbak`)
- `Phase C (dice)/`
  - `Dice.pdsprj`
  - Workspace XML
  - Phase backups
- `Project Backups/`
  - Timestamped backup `.pdsprj` files for restoration/history

**Purpose:** Circuit/system simulation workflow with incremental design phases and many saved checkpoints.

---

## 3) Required Software (Build/Run Environment)

Use the following tools depending on the project:

- **AutoCAD** (or compatible DWG viewer/editor) for `AutoCAD Project IUT`
- **MATLAB (App Designer support required)** for `Matlab GUI project`
- **Python 3 + pip + virtual environment** for `MusicComparison-main`
- **Proteus Design Suite (8.x or compatible)** for `Snake-Ladder game`

---

## 4) Procedures (A–Z Workflow)

Follow this end-to-end process to work with every project in this repository.

### Step A — Clone/Open repository
1. Clone this repository locally.
2. Open the top-level folder in your code editor/file explorer.

### Step B — Decide project domain
- CAD work: go to `AutoCAD Project IUT/`
- MATLAB GUI work: go to `Matlab GUI project/`
- Python audio project: go to `MusicComparison-main/MusicComparison-main/`
- Proteus simulation: go to `Snake-Ladder game/Snake-Ladder game/`

### Step C — Use the right software
- Open `.dwg` in AutoCAD
- Open `.mlapp` in MATLAB App Designer
- Open `.pdsprj` in Proteus
- Use terminal + Python for the music project

### Step D — Python project setup (`MusicComparison-main`)
From `MusicComparison-main/MusicComparison-main/`:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step E — Build data artifacts (Python project)

```bash
python build_metadata.py
python build_database.py
python build_embeddings.py
```

### Step F — Run the Python application

```bash
streamlit run app.py
# or
python live_query.py
```

### Step G — Evaluate model performance

```bash
python evaluate.py
```

### Step H — MATLAB app usage
1. Open `RLC_Analyzer.mlapp` in MATLAB.
2. Run the app.
3. Enter `R`, `L`, `C`, `V`, and `f`.
4. Click **Calculate & Plot**.
5. Read `Z`, `I`, and phase angle, then inspect plotted waveforms.

### Step I — AutoCAD design workflow
1. Open required `.dwg` file.
2. Edit/annotate design as needed.
3. Save a new version or maintain `.bak` backups.
4. Use `4418 Final files/` for final variants and archived versions.

### Step J — Proteus Snake-Ladder workflow
1. Open phase project (`Phase A & B` or `Phase C (dice)`).
2. Simulate and validate logic.
3. Save new snapshots in project backups to preserve history.

### Step K — Final output management
- Keep project-specific outputs inside each project folder.
- Preserve backup files for traceability.
- Use clear naming for new revisions.

---

## 5) Practical Notes

- This repository combines different technology stacks; there is no single unified build command for all folders.
- Most files in CAD/Proteus areas are binary project assets and are edited in domain-specific software.
- For detailed algorithmic explanation of the Python music project, read:
  - `MusicComparison-main/MusicComparison-main/README.md`

---

## 6) Quick Start Summary

1. Pick the project folder.
2. Open it in the correct software toolchain.
3. For the Python project, install dependencies and run `app.py`.
4. For MATLAB/CAD/Proteus, open project files directly and continue design/simulation.
5. Keep backups and versioned outputs organized inside their respective folders.
