# Accelerated Map Matching for GPS Trajectories

A Python replication of the algorithm proposed in:

> Dogramadzi, M., & Khan, A. (2022). *Accelerated Map Matching for GPS Trajectories.*
> IEEE Transactions on Intelligent Transportation Systems, 23(5), 4593–4602.
## Project Structure
map_matching_project/
├── src/
│   ├── road_network.py      # Synthetic road network + segmentation
│   ├── gps_simulator.py     # GPS trajectory simulator with noise
│   └── map_matching.py      # Newson baseline + AMM algorithm
├── plots/                   # Output figures
├── data/                    # Data folder
├── main.py                  # Run experiments and generate plots
├── requirements.txt
└── README.md

## Algorithm Overview

Three key components from the paper are implemented:

- **Newson's HMM baseline** — Hidden Markov Model + Viterbi algorithm
- **AMM (Algorithm 1)** — Replaces expensive routing calculations with
  segment-adjacency heuristic (same / adjacent / non-adjacent penalty)
- **Hybridisation** — Automatically switches between AMM and Newson
  based on GPS point spacing
- **Map Stitching** — Splits journey into sub-windows for large road maps

## Results

| Method | Run-time | Accuracy | Reduction |
|---|---|---|---|
| Newson | 1.812s | 45.5% | — |
| AMM (no stitching) | 1.791s | 42.9% | 1.1% |
| AMM (with stitching) | 0.172s | 92.7% | **90.5%** |

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py
```

## Output Plots

- `plots/fig8_runtime_vs_sampling.png` — Run-time vs sampling rate
- `plots/fig9_accuracy_vs_sampling.png` — Accuracy vs sampling rate
- `plots/runtime_bar.png` — Run-time comparison bar chart
- `plots/trajectory_comparison.png` — Visual trajectory comparison

## Course

M.Tech — International Institute of Information Technology Bangalore (IIITB)
1-month project, June 2026