"""
main.py
-------
Runs all experiments and generates plots replicating
Figures 8 & 9 from Dogramadzi & Khan, IEEE TITS 2022.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from road_network import RoadNetwork
from gps_simulator import TrajectorySimulator, downsample
from map_matching import NewsonMatcher, AMMatcher

# ── Accuracy metric ───────────────────────────────────────────────────────────

def compute_accuracy(gps_points, matched):
    correct, total = 0, 0
    for gp, m in zip(gps_points, matched):
        if m is not None:
            total += 1
            if m.segment.seg_id == gp.true_segment_id:
                correct += 1
    return correct / total if total > 0 else 0.0


# ── Single experiment ─────────────────────────────────────────────────────────

def run_experiment(network, trajectory):
    gps_coords = [p.coords for p in trajectory]
    newson = NewsonMatcher(network)
    amm    = AMMatcher(network)

    matched_n,  t_n  = newson.match(gps_coords)
    matched_ns, t_ns = amm.match(gps_coords, use_stitching=False)
    matched_as, t_as = amm.match(gps_coords, use_stitching=True)

    return {
        "n_points"          : len(trajectory),
        "newson_time"       : t_n,
        "newson_acc"        : compute_accuracy(trajectory, matched_n),
        "amm_no_stitch_time": t_ns,
        "amm_no_stitch_acc" : compute_accuracy(trajectory, matched_ns),
        "amm_stitch_time"   : t_as,
        "amm_stitch_acc"    : compute_accuracy(trajectory, matched_as),
        "matched_newson"    : matched_n,
        "matched_amm"       : matched_as,
        "trajectory"        : trajectory,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_runtime_vs_sampling(sr):
    x   = [1, 0.5, 0.25]
    t_n = [sr[k]['newson_time']     for k in ["1Hz","0.5Hz","0.25Hz"]]
    t_a = [sr[k]['amm_stitch_time'] for k in ["1Hz","0.5Hz","0.25Hz"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, t_n, 'o-', color='#E05A2B', label="Newson's Method")
    ax.plot(x, t_a, 's-', color='#2B7BE0', label="Proposed AMM")
    ax.set_xlabel("GPS Sampling Rate (Hz)")
    ax.set_ylabel("Run-time (seconds)")
    ax.set_title("Fig. 8 — Sampling Rate vs Run-time\n"
                 "(Replication of Dogramadzi & Khan, 2022)")
    ax.set_xticks(x)
    ax.set_xticklabels(["1 Hz", "0.5 Hz", "0.25 Hz"])
    ax.set_xlim(1.1, 0.15)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/fig8_runtime_vs_sampling.png", dpi=150)
    plt.close(fig)
    print("  Saved: plots/fig8_runtime_vs_sampling.png")


def plot_accuracy_vs_sampling(sr):
    x   = [1, 0.5, 0.25]
    a_n = [sr[k]['newson_acc']*100     for k in ["1Hz","0.5Hz","0.25Hz"]]
    a_a = [sr[k]['amm_stitch_acc']*100 for k in ["1Hz","0.5Hz","0.25Hz"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, a_n, 'o-', color='#E05A2B', label="Newson's Method")
    ax.plot(x, a_a, 's-', color='#2B7BE0', label="Proposed AMM")
    ax.set_xlabel("GPS Sampling Rate (Hz)")
    ax.set_ylabel("Accuracy (% correct segment)")
    ax.set_title("Fig. 9 — Sampling Rate vs Accuracy\n"
                 "(Replication of Dogramadzi & Khan, 2022)")
    ax.set_xticks(x)
    ax.set_xticklabels(["1 Hz", "0.5 Hz", "0.25 Hz"])
    ax.set_xlim(1.1, 0.15)
    ax.set_ylim(60, 100)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/fig9_accuracy_vs_sampling.png", dpi=150)
    plt.close(fig)
    print("  Saved: plots/fig9_accuracy_vs_sampling.png")


def plot_runtime_bar(avg):
    methods = ["Newson", "AMM\n(no stitch)", "AMM\n(with stitch)"]
    times   = [avg['newson_time'], avg['amm_no_stitch_time'], avg['amm_stitch_time']]
    colors  = ['#E05A2B', '#7FB3E0', '#2B7BE0']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, times, color=colors, width=0.5, edgecolor='white')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{t:.3f}s", ha='center', va='bottom', fontsize=10)
    for bar, r in zip(bars[1:], [
        (1 - times[1]/times[0])*100,
        (1 - times[2]/times[0])*100
    ]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"-{r:.1f}%", ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')
    ax.set_ylabel("Average Run-time (seconds)")
    ax.set_title("Run-time Comparison at 1 Hz\n"
                 "(Replication of Dogramadzi & Khan, 2022)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/runtime_bar.png", dpi=150)
    plt.close(fig)
    print("  Saved: plots/runtime_bar.png")


def plot_trajectory(network, result):
    traj = result['trajectory'][:120]
    m_n  = result['matched_newson'][:120]
    m_a  = result['matched_amm'][:120]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, matched, title in zip(
        axes, [m_n, m_a],
        ["Newson's Method", "Proposed AMM Method"]
    ):
        for seg in network.segments:
            ax.plot([seg.start[1], seg.end[1]],
                    [seg.start[0], seg.end[0]],
                    color='#BDC3C7', linewidth=0.6, zorder=1)

        gt_lons = [p.true_position[1] for p in traj]
        gt_lats = [p.true_position[0] for p in traj]
        ax.plot(gt_lons, gt_lats, '-', color='#2ECC71',
                linewidth=2.5, label='Ground Truth', zorder=3, alpha=0.8)

        gps_lons = [p.lon for p in traj]
        gps_lats = [p.lat for p in traj]
        ax.scatter(gps_lons, gps_lats, s=6, color='#E74C3C',
                   label='GPS (noisy)', zorder=4, alpha=0.5)

        ml = [m for m in matched if m is not None]
        if ml:
            ax.plot([m.position[1] for m in ml],
                    [m.position[0] for m in ml],
                    '-', color='#2B7BE0', linewidth=1.8,
                    label='Matched', zorder=5, alpha=0.9)

        acc = compute_accuracy(traj, matched)
        ax.set_title(f"{title}  —  Accuracy: {acc*100:.1f}%", fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(fontsize=9)
        ax.set_aspect('equal')

    fig.suptitle("Map Matching: Ground Truth vs Matched Trajectories\n"
                 "(Replication of Dogramadzi & Khan, IEEE TITS 2022)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig("plots/trajectory_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved: plots/trajectory_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MAP MATCHING REPLICATION — Dogramadzi & Khan (2022)")
    print("=" * 60)

    print("\nBuilding road network...")
    network = RoadNetwork(seed=42)
    sim     = TrajectorySimulator(network, seed=42)
    print(f"  Roads: {len(network.roads)}  |  Segments: {len(network.segments)}")

    # ── Experiment 1: 1Hz comparison ─────────────────────────────────────────
    print("\n[Experiment 1] Run-time & Accuracy at 1Hz (5 trajectories)")
    results_1hz = []
    for i in range(5):
        traj = sim.generate(n_points=300, start_seg_idx=i*20)
        res  = run_experiment(network, traj)
        results_1hz.append(res)
        print(f"  Traj {i+1}: "
              f"Newson={res['newson_time']:.3f}s ({res['newson_acc']*100:.1f}%)  "
              f"AMM(no stitch)={res['amm_no_stitch_time']:.3f}s "
              f"({res['amm_no_stitch_acc']*100:.1f}%)  "
              f"AMM(stitch)={res['amm_stitch_time']:.3f}s "
              f"({res['amm_stitch_acc']*100:.1f}%)")

    avg = {k: np.mean([r[k] for r in results_1hz])
           for k in ['newson_time','amm_no_stitch_time','amm_stitch_time',
                     'newson_acc','amm_no_stitch_acc','amm_stitch_acc']}

    print(f"\n  {'Method':<25} {'Time(s)':>8}  {'Accuracy':>9}  {'Reduction':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Newson':<25} {avg['newson_time']:>8.3f}  "
          f"{avg['newson_acc']*100:>8.1f}%  {'—':>10}")
    print(f"  {'AMM (no stitching)':<25} {avg['amm_no_stitch_time']:>8.3f}  "
          f"{avg['amm_no_stitch_acc']*100:>8.1f}%  "
          f"{(1-avg['amm_no_stitch_time']/avg['newson_time'])*100:>9.1f}%")
    print(f"  {'AMM (with stitching)':<25} {avg['amm_stitch_time']:>8.3f}  "
          f"{avg['amm_stitch_acc']*100:>8.1f}%  "
          f"{(1-avg['amm_stitch_time']/avg['newson_time'])*100:>9.1f}%")

    # ── Experiment 2: Sampling rate ───────────────────────────────────────────
    print("\n[Experiment 2] Accuracy & Run-time vs Sampling Rate")
    long_traj = sim.generate(n_points=600, start_seg_idx=10)
    sr = {}
    for label, traj in [("1Hz",   long_traj),
                         ("0.5Hz", downsample(long_traj, 2)),
                         ("0.25Hz",downsample(long_traj, 4))]:
        res = run_experiment(network, traj)
        sr[label] = res
        print(f"  {label:6s} | N={res['n_points']:4d} | "
              f"Newson: {res['newson_time']:.3f}s {res['newson_acc']*100:.1f}% | "
              f"AMM:    {res['amm_stitch_time']:.3f}s "
              f"{res['amm_stitch_acc']*100:.1f}%")

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    os.makedirs("plots", exist_ok=True)
    plot_runtime_vs_sampling(sr)
    plot_accuracy_vs_sampling(sr)
    plot_runtime_bar(avg)
    plot_trajectory(network, results_1hz[0])

    print("\n" + "=" * 60)
    print("  Done! Check the plots/ folder for output figures.")
    print("=" * 60)