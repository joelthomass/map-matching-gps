"""
main_real.py
------------
Runs the map matching experiments on a REAL road network
fetched from OpenStreetMap (Koramangala, Bengaluru).
This should reproduce a meaningful run-time difference
between Newson and AMM since routing on a real network
is expensive.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math

from real_road_network import RealRoadNetwork
from road_network import (
    Segment, CandidatePoint, euclidean_distance,
    closest_point_on_segment, MAX_SEGMENT_LENGTH
)
from gps_simulator import GPSPoint
from map_matching import (
    AMMatcher, viterbi, CANDIDATE_RADIUS,
    measurement_probability, transition_probability
)
import numpy as np


# ── GPS Simulator for real network ───────────────────────────────────────────

SIGMA_GPS = 4.07
SPEED_MS  = 8.0    # ~30 km/h, slower for urban area


def add_noise(position, sigma_m, rng):
    noise_lat = rng.normal(0, sigma_m)
    noise_lon = rng.normal(0, sigma_m)
    cos_lat = math.cos(math.radians(position[0]))
    return (
        position[0] + noise_lat / 111_320.0,
        position[1] + noise_lon / (111_320.0 * cos_lat)
    )


def interpolate(seg, t):
    return (
        seg.start[0] + t * (seg.end[0] - seg.start[0]),
        seg.start[1] + t * (seg.end[1] - seg.start[1])
    )


def generate_trajectory(network, n_points=300, seed=0):
    """Simulate a vehicle driving on the real road network."""
    rng = np.random.default_rng(seed)

    # Pick a random starting segment
    start_idx = int(rng.integers(0, len(network.segments)))
    current_seg = network.segments[start_idx]
    t = 0.0
    step_m = SPEED_MS   # 1 Hz sampling

    gps_points = []
    for _ in range(n_points):
        true_pos = interpolate(current_seg, t)
        noisy_pos = add_noise(true_pos, SIGMA_GPS, rng)
        gps_points.append(GPSPoint(
            lat=noisy_pos[0],
            lon=noisy_pos[1],
            true_segment_id=current_seg.seg_id,
            true_position=true_pos
        ))

        advance = step_m / max(current_seg.length, 1.0)
        t += advance

        while t >= 1.0:
            t -= 1.0
            adj_ids = network._adjacency.get(current_seg.seg_id, [])
            if not adj_ids:
                t = 0.0
                break
            next_id = int(rng.choice(adj_ids))
            next_seg = network.segments[next_id]
            dist_start = euclidean_distance(
                next_seg.start, current_seg.end)
            dist_end = euclidean_distance(
                next_seg.end, current_seg.end)
            if dist_start < dist_end:
                current_seg = next_seg
            else:
                current_seg = Segment(
                    seg_id=next_seg.seg_id,
                    road_id=next_seg.road_id,
                    start=next_seg.end,
                    end=next_seg.start,
                    length=next_seg.length
                )

    return gps_points


def downsample(traj, factor):
    return traj[::factor]


# ── Fair Newson with stitching ────────────────────────────────────────────────

class NewsonMatcherStitched:
    def __init__(self, network, candidate_radius=CANDIDATE_RADIUS,
                 stitch_size=50):
        self.network = network
        self.candidate_radius = candidate_radius
        self.stitch_size = stitch_size

    def _route_distance(self, ci, cj):
        return self.network.routing_distance(
            ci.position, cj.position, ci.segment, cj.segment)

    def _get_candidates(self, gps_points, seg_list):
        all_candidates = []
        for gp in gps_points:
            cands = []
            for seg in seg_list:
                closest, dist = closest_point_on_segment(
                    gp, seg.start, seg.end)
                if dist <= self.candidate_radius:
                    cands.append(CandidatePoint(
                        segment=seg,
                        position=closest,
                        distance_to_gps=dist))
            all_candidates.append(cands)
        return all_candidates

    def _match_window(self, gps_points, seg_list):
        all_candidates = self._get_candidates(gps_points, seg_list)
        return viterbi(gps_points, all_candidates, self._route_distance)

    def match(self, gps_points, use_stitching=True):
        t0 = time.perf_counter()

        if not use_stitching or len(gps_points) <= self.stitch_size:
            seg_list = self.network.segments
            matched = self._match_window(gps_points, seg_list)
            return matched, time.perf_counter() - t0

        all_matched = []
        step = self.stitch_size
        i = 0
        while i < len(gps_points):
            window = gps_points[i: i + step + 1]
            sub_segs = self.network.get_subnetwork_segments(
                window, buffer_m=100.0)
            window_matched = self._match_window(window, sub_segs)
            if i + step + 1 >= len(gps_points):
                all_matched.extend(window_matched)
            else:
                all_matched.extend(window_matched[:-1])
            i += step

        return all_matched, time.perf_counter() - t0


# ── Accuracy ──────────────────────────────────────────────────────────────────

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
    newson = NewsonMatcherStitched(network)
    amm    = AMMatcher(network)

    matched_n, t_n = newson.match(gps_coords, use_stitching=True)
    matched_a, t_a = amm.match(gps_coords,   use_stitching=True)

    return {
        "n_points"   : len(trajectory),
        "newson_time": t_n,
        "newson_acc" : compute_accuracy(trajectory, matched_n),
        "amm_time"   : t_a,
        "amm_acc"    : compute_accuracy(trajectory, matched_a),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_runtime_vs_sampling(sr, out):
    x   = [1, 0.5, 0.25]
    t_n = [sr[k]['newson_time'] for k in ["1Hz", "0.5Hz", "0.25Hz"]]
    t_a = [sr[k]['amm_time']   for k in ["1Hz", "0.5Hz", "0.25Hz"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, t_n, 'o-', color='#E05A2B', label="Newson's Method")
    ax.plot(x, t_a, 's-', color='#2B7BE0', label="Proposed AMM")
    ax.set_xlabel("GPS Sampling Rate (Hz)")
    ax.set_ylabel("Run-time (seconds)")
    ax.set_title("GPS Sampling Rate vs Run-time\n"
                 "Real Road Network — Koramangala, Bengaluru")
    ax.set_xticks(x)
    ax.set_xticklabels(["1 Hz", "0.5 Hz", "0.25 Hz"])
    ax.set_xlim(1.1, 0.15)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved:", out)


def plot_accuracy_vs_sampling(sr, out):
    x   = [1, 0.5, 0.25]
    a_n = [sr[k]['newson_acc'] * 100 for k in ["1Hz", "0.5Hz", "0.25Hz"]]
    a_a = [sr[k]['amm_acc']   * 100 for k in ["1Hz", "0.5Hz", "0.25Hz"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, a_n, 'o-', color='#E05A2B', label="Newson's Method")
    ax.plot(x, a_a, 's-', color='#2B7BE0', label="Proposed AMM")
    ax.set_xlabel("GPS Sampling Rate (Hz)")
    ax.set_ylabel("Accuracy (% correct segment)")
    ax.set_title("GPS Sampling Rate vs Accuracy\n"
                 "Real Road Network — Koramangala, Bengaluru")
    ax.set_xticks(x)
    ax.set_xticklabels(["1 Hz", "0.5 Hz", "0.25 Hz"])
    ax.set_xlim(1.1, 0.15)
    ax.set_ylim(40, 100)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved:", out)


def plot_runtime_bar(avg, out):
    methods = ["Newson\n(with stitching)", "AMM\n(with stitching)"]
    times   = [avg['newson_time'], avg['amm_time']]
    colors  = ['#E05A2B', '#2B7BE0']
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(methods, times, color=colors, width=0.4, edgecolor='white')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{t:.3f}s", ha='center', va='bottom', fontsize=11)
    if times[0] > 0:
        red = (1 - times[1] / times[0]) * 100
        color = 'white' if red > 0 else 'black'
        ax.text(bars[1].get_x() + bars[1].get_width() / 2,
                times[1] / 2,
                f"{red:+.1f}%", ha='center', va='center',
                fontsize=11, color=color, fontweight='bold')
    ax.set_ylabel("Average Run-time (seconds)")
    ax.set_title("Run-time Comparison at 1 Hz\n"
                 "Real Road Network — Koramangala, Bengaluru")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved:", out)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  REAL ROAD NETWORK EXPERIMENT")
    print("  Koramangala, Bengaluru — OSM data")
    print("=" * 60)

    print("\nLoading road network...")
    network = RealRoadNetwork(
        center_point=(12.9352, 77.6245),
        dist_m=1000.0
    )

    os.makedirs("plots", exist_ok=True)

    # ── Experiment 1: 1Hz, 3 trajectories ────────────────────────────────────
    print("\n[Experiment 1] Run-time & Accuracy at 1Hz (3 trajectories)\n")
    results_1hz = []
    for i in range(3):
        print(f"  Running trajectory {i+1}/3...")
        traj = generate_trajectory(network, n_points=200, seed=i*10)
        res  = run_experiment(network, traj)
        results_1hz.append(res)
        print(f"  Traj {i+1}: "
              f"Newson={res['newson_time']:.3f}s ({res['newson_acc']*100:.1f}%)  "
              f"AMM={res['amm_time']:.3f}s ({res['amm_acc']*100:.1f}%)")

    avg = {k: np.mean([r[k] for r in results_1hz])
           for k in ['newson_time', 'amm_time', 'newson_acc', 'amm_acc']}

    print(f"\n  AVERAGES:")
    print(f"  {'Method':<30} {'Time(s)':>8}  {'Accuracy':>9}  {'Reduction':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Newson (with stitching)':<30} {avg['newson_time']:>8.3f}  "
          f"{avg['newson_acc']*100:>8.1f}%  {'—':>10}")
    t_red = (1 - avg['amm_time'] / avg['newson_time']) * 100
    print(f"  {'AMM (with stitching)':<30} {avg['amm_time']:>8.3f}  "
          f"{avg['amm_acc']*100:>8.1f}%  {t_red:>9.1f}%")

    # ── Experiment 2: Sampling rates ──────────────────────────────────────────
    print("\n[Experiment 2] Accuracy & Run-time vs Sampling Rate\n")
    long_traj = generate_trajectory(network, n_points=400, seed=99)
    sr = {}
    for label, traj in [("1Hz",    long_traj),
                         ("0.5Hz",  downsample(long_traj, 2)),
                         ("0.25Hz", downsample(long_traj, 4))]:
        print(f"  Running {label}...")
        res = run_experiment(network, traj)
        sr[label] = res
        print(f"  {label:6s} | N={res['n_points']:4d} | "
              f"Newson: {res['newson_time']:.3f}s {res['newson_acc']*100:.1f}% | "
              f"AMM: {res['amm_time']:.3f}s {res['amm_acc']*100:.1f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_runtime_vs_sampling(sr, "plots/fig8_runtime_vs_sampling.png")
    plot_accuracy_vs_sampling(sr, "plots/fig9_accuracy_vs_sampling.png")
    plot_runtime_bar(avg, "plots/runtime_bar.png")

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print(f"  Network: {len(network.segments)} segments (real OSM)")
    print(f"  Newson : {avg['newson_time']:.3f}s  {avg['newson_acc']*100:.1f}%")
    print(f"  AMM    : {avg['amm_time']:.3f}s  {avg['amm_acc']*100:.1f}%")
    print(f"  Run-time reduction: {t_red:.1f}%")
    print("=" * 60)