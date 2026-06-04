"""
real_road_network.py
--------------------
Builds a real road network from OpenStreetMap using OSMnx,
then segments it exactly as in the paper (<=60m segments).
Replaces the synthetic RoadNetwork for experiments.
"""

import math
import osmnx as ox
import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from road_network import (
    Segment, CandidatePoint, Road,
    euclidean_distance, closest_point_on_segment,
    MAX_SEGMENT_LENGTH
)


class RealRoadNetwork:
    """
    Road network built from OpenStreetMap data via OSMnx.
    Interface is identical to RoadNetwork so all existing
    map matching code works without any changes.
    """

    def __init__(
        self,
        center_point: Tuple[float, float] = (12.9352, 77.6245),
        dist_m: float = 1000.0,
        network_type: str = "drive",
        seed: int = 42
    ):
        self.center_point = center_point
        self.rng = np.random.default_rng(seed)

        print(f"  Fetching road network from OSM "
              f"({dist_m}m radius around {center_point})...")
        self._G = ox.graph_from_point(
            center_point, dist=dist_m, network_type=network_type
        )
        # Project to UTM for accurate metre-based distances
        self._G_proj = ox.project_graph(self._G)

        self.roads: List[Road] = []
        self.segments: List[Segment] = []
        self._seg_counter = 0
        self._road_counter = 0
        self._adjacency: Dict[int, List[int]] = {}

        self._build_network()
        self._build_adjacency()

        print(f"  OSM graph: {len(self._G.nodes)} nodes, "
              f"{len(self._G.edges)} edges")
        print(f"  After segmentation: {len(self.segments)} segments")

    def _build_network(self):
        """Convert OSMnx edges to Road/Segment objects."""
        for u, v, data in self._G.edges(data=True):
            # Get node coordinates (lat, lon)
            u_data = self._G.nodes[u]
            v_data = self._G.nodes[v]
            start = (u_data['y'], u_data['x'])   # (lat, lon)
            end   = (v_data['y'], v_data['x'])

            road = Road(
                road_id=self._road_counter,
                start=start,
                end=end
            )
            self._road_counter += 1
            road.segments = self._segment_road(road)
            self.roads.append(road)
            self.segments.extend(road.segments)

    def _segment_road(self, road: Road) -> List[Segment]:
        length = euclidean_distance(road.start, road.end)
        if length < 1.0:   # skip zero-length edges
            return []
        return self._halve(road.road_id, road.start, road.end, length)

    def _halve(self, road_id, a, b, length):
        if length <= MAX_SEGMENT_LENGTH:
            seg = Segment(
                seg_id=self._seg_counter,
                road_id=road_id,
                start=a, end=b,
                length=length
            )
            self._seg_counter += 1
            return [seg]
        mid = ((a[0]+b[0])/2, (a[1]+b[1])/2)
        return (self._halve(road_id, a, mid, length/2) +
                self._halve(road_id, mid, b, length/2))

    def _build_adjacency(self):
        """Two segments are adjacent if they share an endpoint."""
        endpoint_map: Dict[Tuple[int,int], List[int]] = {}

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

    # ── Public API (identical to RoadNetwork) ────────────────────────────────

    def are_adjacent(self, seg_a: Segment, seg_b: Segment) -> bool:
        return seg_b.seg_id in self._adjacency.get(seg_a.seg_id, [])

    def same_segment(self, seg_a: Segment, seg_b: Segment) -> bool:
        return seg_a.seg_id == seg_b.seg_id

    def get_candidates(
        self,
        gps_point: Tuple[float, float],
        radius_m: float = 50.0
    ) -> List[CandidatePoint]:
        candidates = []
        for seg in self.segments:
            closest, dist = closest_point_on_segment(
                gps_point, seg.start, seg.end)
            if dist <= radius_m:
                candidates.append(CandidatePoint(
                    segment=seg,
                    position=closest,
                    distance_to_gps=dist
                ))
        return candidates

    def routing_distance(self, p1, p2, seg1, seg2) -> float:
        """
        True road-network routing distance using NetworkX shortest path.
        This is what makes Newson expensive on a real network.
        """
        # Find nearest OSM nodes to each candidate point
        n1 = ox.distance.nearest_nodes(
            self._G, X=p1[1], Y=p1[0])
        n2 = ox.distance.nearest_nodes(
            self._G, X=p2[1], Y=p2[0])
        try:
            length = nx.shortest_path_length(
                self._G, n1, n2, weight='length')
            return length
        except nx.NetworkXNoPath:
            # No path found — return large value
            return euclidean_distance(p1, p2) + 1000.0

    def get_subnetwork_segments(
        self,
        gps_points: List[Tuple[float, float]],
        buffer_m: float = 100.0
    ) -> List[Segment]:
        """Return only segments near the given GPS points."""
        relevant = []
        for seg in self.segments:
            mid = seg.midpoint
            for gp in gps_points:
                if euclidean_distance(gp, mid) <= buffer_m:
                    relevant.append(seg)
                    break
        return relevant