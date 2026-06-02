"""
map_matching.py
---------------
Implements two map matching algorithms:
1. NewsonMatcher  - baseline HMM + Viterbi (paper Section II-B)
2. AMMatcher      - Accelerated Map Matching with hybridisation
                    and Map Stitching (paper Section III)
"""

import math
import time
from typing import List, Optional
from dataclasses import dataclass

from road_network import (
    RoadNetwork, Segment, CandidatePoint,
    euclidean_distance, closest_point_on_segment, MAX_SEGMENT_LENGTH
)
from gps_simulator import GPSPoint

# Paper parameters
SIGMA            = 4.07
BETA             = 17.64
CANDIDATE_RADIUS = 50.0
LARGE_PENALTY    = 1000.0
MAP_STITCH_GROUP = 50


def measurement_probability(distance_m, sigma=SIGMA):
    """Mp = (1/sigma) * exp(-De/sigma)  — paper Section II-B"""
    return (1.0 / sigma) * math.exp(-distance_m / sigma)


def transition_probability(dt, beta=BETA):
    """Tp = (1/beta) * exp(-|Dr-De|/beta)  — paper Section II-B"""
    return (1.0 / beta) * math.exp(-dt / beta)


@dataclass
class ViterbiState:
    prob: float
    candidate: CandidatePoint
    prev_idx: int


def viterbi(gps_points, all_candidates, route_dist_fn):
    """Viterbi algorithm over HMM of candidate points — paper Section II-B-2"""
    N = len(gps_points)
    if N == 0:
        return []

    dp = []

    # Step 0: initialise with measurement probability only
    first_cands = all_candidates[0]
    if not first_cands:
        return [None] * N

    dp.append([
        ViterbiState(prob=measurement_probability(c.distance_to_gps),
                     candidate=c, prev_idx=-1)
        for c in first_cands
    ])

    # Steps 1 to N-1
    for i in range(1, N):
        curr_cands = all_candidates[i]
        prev_states = dp[i-1]

        if not curr_cands:
            dp.append([])
            continue

        curr_states = []
        for c_curr in curr_cands:
            mp = measurement_probability(c_curr.distance_to_gps)
            best_prob, best_prev = -1.0, 0

            for k, prev_state in enumerate(prev_states):
                c_prev = prev_state.candidate
                de = euclidean_distance(c_curr.position, c_prev.position)
                dr = route_dist_fn(c_curr, c_prev)
                dt = abs(dr - de)
                tp = transition_probability(dt)
                prob = prev_state.prob * mp * tp
                if prob > best_prob:
                    best_prob, best_prev = prob, k

            curr_states.append(ViterbiState(prob=best_prob,
                                            candidate=c_curr,
                                            prev_idx=best_prev))
        dp.append(curr_states)

    # Backtrack
    matched = [None] * N
    if not dp[N-1]:
        return matched

    best_final = max(range(len(dp[N-1])), key=lambda k: dp[N-1][k].prob)
    matched[N-1] = dp[N-1][best_final].candidate

    idx = best_final
    for i in range(N-2, -1, -1):
        if not dp[i]:
            break
        idx = dp[i+1][idx].prev_idx if idx < len(dp[i+1]) else 0
        if idx < 0 or idx >= len(dp[i]):
            idx = 0
        matched[i] = dp[i][idx].candidate

    return matched


# ── Newson Baseline ───────────────────────────────────────────────────────────

class NewsonMatcher:
    """HMM map matching using actual routing distance for Tp."""

    def __init__(self, network: RoadNetwork,
                 candidate_radius: float = CANDIDATE_RADIUS):
        self.network = network
        self.candidate_radius = candidate_radius

    def _route_distance(self, ci, cj):
        return self.network.routing_distance(
            ci.position, cj.position, ci.segment, cj.segment)

    def _get_candidates(self, gps_points, seg_list):
        all_candidates = []
        for gp in gps_points:
            cands = []
            for seg in seg_list:
                closest, dist = closest_point_on_segment(gp, seg.start, seg.end)
                if dist <= self.candidate_radius:
                    cands.append(CandidatePoint(segment=seg, position=closest,
                                                distance_to_gps=dist))
            all_candidates.append(cands)
        return all_candidates

    def match(self, gps_points, segments=None):
        """Returns (matched_candidates, run_time_seconds)"""
        t0 = time.perf_counter()
        seg_list = segments if segments is not None else self.network.segments
        all_candidates = self._get_candidates(gps_points, seg_list)
        matched = viterbi(gps_points, all_candidates, self._route_distance)
        return matched, time.perf_counter() - t0


# ── Accelerated Map Matching ──────────────────────────────────────────────────

class AMMatcher:
    """
    Accelerated Map Matching.
    A) Segment-adjacency route distance approximation (Algorithm 1)
    B) Hybridisation: falls back to Newson when GPS points are far apart
    C) Map Stitching: processes journey in overlapping sub-windows
    """

    def __init__(self, network: RoadNetwork,
                 candidate_radius: float = CANDIDATE_RADIUS,
                 stitch_size: int = MAP_STITCH_GROUP):
        self.network = network
        self.candidate_radius = candidate_radius
        self.stitch_size = stitch_size

    def _amm_route_distance(self, ci, cj):
        """
        Algorithm 1 — approximated route distance using adjacency.
        Same segment    → Euclidean only
        Adjacent segs   → Euclidean + small penalty (Πs)
        Non-adjacent    → Euclidean + large penalty (Πl)
        """
        de = euclidean_distance(ci.position, cj.position)
        if self.network.same_segment(ci.segment, cj.segment):
            return de
        elif self.network.are_adjacent(ci.segment, cj.segment):
            penalty = (ci.segment.length + cj.segment.length) / 2.0
            return de + penalty
        else:
            return de + LARGE_PENALTY

    def _use_accelerated(self, gp_curr, gp_prev):
        """
        Hybridisation check — use AMM only when GPS points
        are close enough (within MAX_SEGMENT_LENGTH).
        Paper: Section III-B
        """
        return euclidean_distance(gp_curr, gp_prev) < MAX_SEGMENT_LENGTH

    def _get_candidates(self, gps_points, seg_list):
        all_candidates = []
        for gp in gps_points:
            cands = []
            for seg in seg_list:
                closest, dist = closest_point_on_segment(gp, seg.start, seg.end)
                if dist <= self.candidate_radius:
                    cands.append(CandidatePoint(segment=seg, position=closest,
                                                distance_to_gps=dist))
            all_candidates.append(cands)
        return all_candidates

    def _viterbi_hybrid(self, gps_points, all_candidates, use_accel_flags):
        """Viterbi that switches route-distance function per step."""
        N = len(gps_points)
        if N == 0:
            return []

        dp = []
        if not all_candidates[0]:
            return [None] * N

        dp.append([
            ViterbiState(prob=measurement_probability(c.distance_to_gps),
                         candidate=c, prev_idx=-1)
            for c in all_candidates[0]
        ])

        for i in range(1, N):
            curr_cands = all_candidates[i]
            prev_states = dp[i-1]
            use_accel = use_accel_flags[i]

            if not curr_cands:
                dp.append([])
                continue

            curr_states = []
            for c_curr in curr_cands:
                mp = measurement_probability(c_curr.distance_to_gps)
                best_prob, best_prev = -1.0, 0

                for k, prev_state in enumerate(prev_states):
                    c_prev = prev_state.candidate
                    de = euclidean_distance(c_curr.position, c_prev.position)
                    if use_accel:
                        dr = self._amm_route_distance(c_curr, c_prev)
                    else:
                        dr = self.network.routing_distance(
                            c_curr.position, c_prev.position,
                            c_curr.segment, c_prev.segment)
                    dt = abs(dr - de)
                    tp = transition_probability(dt)
                    prob = prev_state.prob * mp * tp
                    if prob > best_prob:
                        best_prob, best_prev = prob, k

                curr_states.append(ViterbiState(prob=best_prob,
                                                candidate=c_curr,
                                                prev_idx=best_prev))
            dp.append(curr_states)

        # Backtrack
        matched = [None] * N
        if not dp[N-1]:
            return matched

        best_final = max(range(len(dp[N-1])), key=lambda k: dp[N-1][k].prob)
        matched[N-1] = dp[N-1][best_final].candidate

        idx = best_final
        for i in range(N-2, -1, -1):
            if not dp[i]:
                break
            idx = dp[i+1][idx].prev_idx if idx < len(dp[i+1]) else 0
            if idx < 0 or idx >= len(dp[i]):
                idx = 0
            matched[i] = dp[i][idx].candidate

        return matched

    def _match_window(self, gps_points, segments=None):
        seg_list = segments if segments is not None else self.network.segments
        all_candidates = self._get_candidates(gps_points, seg_list)

        use_accel_flags = [True]
        for i in range(1, len(gps_points)):
            use_accel_flags.append(
                self._use_accelerated(gps_points[i], gps_points[i-1]))

        return self._viterbi_hybrid(gps_points, all_candidates, use_accel_flags)

    def match(self, gps_points, use_stitching=True):
        """Returns (matched_candidates, run_time_seconds)"""
        t0 = time.perf_counter()

        if not use_stitching or len(gps_points) <= self.stitch_size:
            matched = self._match_window(gps_points)
            return matched, time.perf_counter() - t0

        # Map Stitching — paper Section III-C
        all_matched = []
        step = self.stitch_size
        i = 0
        while i < len(gps_points):
            window = gps_points[i: i + step + 1]
            sub_segs = self.network.get_subnetwork_segments(window, buffer_m=100.0)
            window_matched = self._match_window(window, sub_segs)

            if i + step + 1 >= len(gps_points):
                all_matched.extend(window_matched)
            else:
                all_matched.extend(window_matched[:-1])
            i += step

        return all_matched, time.perf_counter() - t0