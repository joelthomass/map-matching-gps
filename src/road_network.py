"""
road_network.py
---------------
Builds a synthetic road network that mimics a real urban grid.
Each road is stored as a sequence of (lat, lon) waypoints.
Segmentation splits every road into sub-segments <= 60m,
exactly as described in Section III-A of the paper.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

MAX_SEGMENT_LENGTH = 60.0
MIN_SEGMENT_LENGTH = 30.0

def haversine(p1, p2):
    R = 6_371_000.0
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def euclidean_distance(p1, p2):
    lat_m = (p2[0] - p1[0]) * 111_320.0
    lon_m = (p2[1] - p1[1]) * 111_320.0 * math.cos(math.radians((p1[0]+p2[0])/2))
    return math.sqrt(lat_m**2 + lon_m**2)

def closest_point_on_segment(p, a, b):
    def to_xy(pt, ref):
        x = (pt[1] - ref[1]) * 111_320.0 * math.cos(math.radians(ref[0]))
        y = (pt[0] - ref[0]) * 111_320.0
        return np.array([x, y])
    ref = a
    A = to_xy(a, ref)
    B = to_xy(b, ref)
    P = to_xy(p, ref)
    AB = B - A
    ab_len_sq = float(np.dot(AB, AB))
    if ab_len_sq < 1e-10:
        return a, euclidean_distance(p, a)
    t = float(np.dot(P - A, AB)) / ab_len_sq
    t = max(0.0, min(1.0, t))
    closest_xy = A + t * AB
    cos_lat = math.cos(math.radians(ref[0]))
    closest_lat = ref[0] + closest_xy[1] / 111_320.0
    closest_lon = ref[1] + closest_xy[0] / (111_320.0 * cos_lat)
    return (closest_lat, closest_lon), euclidean_distance(p, (closest_lat, closest_lon))

@dataclass
class Segment:
    seg_id: int
    road_id: int
    start: Tuple[float, float]
    end:   Tuple[float, float]
    length: float

    @property
    def midpoint(self):
        return ((self.start[0]+self.end[0])/2, (self.start[1]+self.end[1])/2)

@dataclass
class CandidatePoint:
    segment: 'Segment'
    position: Tuple[float, float]
    distance_to_gps: float

@dataclass
class Road:
    road_id: int
    start: Tuple[float, float]
    end:   Tuple[float, float]
    segments: List['Segment'] = field(default_factory=list)

class RoadNetwork:
    def __init__(self, origin=(47.606, -122.332), n_rows=10, n_cols=10,
                 spacing_m=150.0, seed=42):
        self.origin = origin
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.spacing_m = spacing_m
        self.rng = np.random.default_rng(seed)
        self.roads: List[Road] = []
        self.segments: List[Segment] = []
        self._seg_counter = 0
        self._road_counter = 0
        self._adjacency: Dict[int, List[int]] = {}
        self._build_network()
        self._build_adjacency()

    def _metres_to_latlon(self, dy_m, dx_m):
        lat = self.origin[0] + dy_m / 111_320.0
        lon = self.origin[1] + dx_m / (111_320.0 * math.cos(math.radians(self.origin[0])))
        return (lat, lon)

    def _grid_point(self, row, col):
        return self._metres_to_latlon(row * self.spacing_m, col * self.spacing_m)

    def _add_road(self, start, end):
        road = Road(road_id=self._road_counter, start=start, end=end)
        self._road_counter += 1
        road.segments = self._segment_road(road)
        self.roads.append(road)
        self.segments.extend(road.segments)

    def _segment_road(self, road):
        total_len = euclidean_distance(road.start, road.end)
        return self._halve(road.road_id, road.start, road.end, total_len)

    def _halve(self, road_id, a, b, length):
        if length <= MAX_SEGMENT_LENGTH:
            seg = Segment(seg_id=self._seg_counter, road_id=road_id,
                          start=a, end=b, length=length)
            self._seg_counter += 1
            return [seg]
        mid = ((a[0]+b[0])/2, (a[1]+b[1])/2)
        return (self._halve(road_id, a, mid, length/2) +
                self._halve(road_id, mid, b, length/2))

    def _build_network(self):
        for r in range(self.n_rows):
            for c in range(self.n_cols - 1):
                self._add_road(self._grid_point(r, c), self._grid_point(r, c+1))
        for r in range(self.n_rows - 1):
            for c in range(self.n_cols):
                self._add_road(self._grid_point(r, c), self._grid_point(r+1, c))
        for r in range(0, self.n_rows-1, 2):
            for c in range(0, self.n_cols-1, 2):
                self._add_road(self._grid_point(r, c), self._grid_point(r+1, c+1))

    def _build_adjacency(self):
        endpoint_map = {}
        def snap(pt):
            return (round(pt[0]*1_000_000), round(pt[1]*1_000_000))
        for seg in self.segments:
            for pt in (seg.start, seg.end):
                endpoint_map.setdefault(snap(pt), []).append(seg.seg_id)
        for seg in self.segments:
            neighbours = []
            for pt in (seg.start, seg.end):
                for other_id in endpoint_map.get(snap(pt), []):
                    if other_id != seg.seg_id:
                        neighbours.append(other_id)
            self._adjacency[seg.seg_id] = list(set(neighbours))

    def are_adjacent(self, seg_a, seg_b):
        return seg_b.seg_id in self._adjacency.get(seg_a.seg_id, [])

    def same_segment(self, seg_a, seg_b):
        return seg_a.seg_id == seg_b.seg_id

    def get_candidates(self, gps_point, radius_m=50.0):
        candidates = []
        for seg in self.segments:
            closest, dist = closest_point_on_segment(gps_point, seg.start, seg.end)
            if dist <= radius_m:
                candidates.append(CandidatePoint(segment=seg, position=closest,
                                                  distance_to_gps=dist))
        return candidates

    def routing_distance(self, p1, p2, seg1, seg2):
        return euclidean_distance(p1, p2)

    def get_subnetwork_segments(self, gps_points, buffer_m=100.0):
        relevant = []
        for seg in self.segments:
            mid = seg.midpoint
            for gp in gps_points:
                if euclidean_distance(gp, mid) <= buffer_m:
                    relevant.append(seg)
                    break
        return relevant