"""
gps_simulator.py
----------------
Generates synthetic GPS trajectories by:
1. Simulating a vehicle driving along roads in the network
2. Adding zero-mean Gaussian noise (sigma = 4.07m) to each position
   as per Newson's GPS error model (paper Section II-B-1)
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from road_network import RoadNetwork, Segment, euclidean_distance

SIGMA_GPS   = 4.07
SPEED_MS    = 13.4
SAMPLE_RATE = 1.0

@dataclass
class GPSPoint:
    lat: float
    lon: float
    true_segment_id: int
    true_position: tuple

    @property
    def coords(self):
        return (self.lat, self.lon)

def _add_gps_noise(position, sigma_m, rng):
    noise_lat_m = rng.normal(0, sigma_m)
    noise_lon_m = rng.normal(0, sigma_m)
    cos_lat = math.cos(math.radians(position[0]))
    noisy_lat = position[0] + noise_lat_m / 111_320.0
    noisy_lon = position[1] + noise_lon_m / (111_320.0 * cos_lat)
    return (noisy_lat, noisy_lon)

def _interpolate_along_segment(seg, t):
    lat = seg.start[0] + t * (seg.end[0] - seg.start[0])
    lon = seg.start[1] + t * (seg.end[1] - seg.start[1])
    return (lat, lon)

class TrajectorySimulator:
    def __init__(self, network: RoadNetwork, seed: int = 0):
        self.network = network
        self.rng = np.random.default_rng(seed)

    def _pick_next_segment(self, current_seg):
        adj_ids = self.network._adjacency.get(current_seg.seg_id, [])
        if not adj_ids:
            return None
        next_id = self.rng.choice(adj_ids)
        return self.network.segments[next_id]

    def generate(self, n_points=300, sigma_m=SIGMA_GPS,
                 start_seg_idx=None, sampling_rate=SAMPLE_RATE):
        if start_seg_idx is None:
            start_seg_idx = int(self.rng.integers(0, len(self.network.segments)))

        current_seg = self.network.segments[start_seg_idx]
        t = 0.0
        step_m = SPEED_MS / sampling_rate
        gps_points = []

        for _ in range(n_points):
            true_pos = _interpolate_along_segment(current_seg, t)
            noisy_pos = _add_gps_noise(true_pos, sigma_m, self.rng)
            gps_points.append(GPSPoint(
                lat=noisy_pos[0], lon=noisy_pos[1],
                true_segment_id=current_seg.seg_id,
                true_position=true_pos
            ))

            advance_fraction = step_m / current_seg.length
            t += advance_fraction

            while t >= 1.0:
                t -= 1.0
                next_seg = self._pick_next_segment(current_seg)
                if next_seg is None:
                    current_seg = Segment(
                        seg_id=current_seg.seg_id, road_id=current_seg.road_id,
                        start=current_seg.end, end=current_seg.start,
                        length=current_seg.length)
                else:
                    dist_start = euclidean_distance(next_seg.start, current_seg.end)
                    dist_end = euclidean_distance(next_seg.end, current_seg.end)
                    if dist_start < dist_end:
                        current_seg = next_seg
                    else:
                        current_seg = Segment(
                            seg_id=next_seg.seg_id, road_id=next_seg.road_id,
                            start=next_seg.end, end=next_seg.start,
                            length=next_seg.length)
        return gps_points

def downsample(trajectory, factor):
    return trajectory[::factor]