"""Unit coverage for TrackingSystem._resolve_cold_conflicts (src/controller.py).

Exercises the conflict-graph resolution used by update_cold_batch to arbitrate
between simultaneously-solved cold-start candidates, without needing a pool,
images, or blob detection — candidate solution dicts are constructed directly,
matching the shape ControllerTracker._compute_fused_solution produces
(primary_cam, error, assignment, aux_assignments).

Shared-blob conflicts are decided by actual blob pixel-position overlap (see
_resolve_cold_conflicts' blob_geometry parameter), not by raw blob-index
equality — main.py detects blobs per controller, so two controllers' blob
index 5 in the same camera are, in general, indices into two independent
arrays, not the same physical blob. _geo() below builds synthetic per-
controller (centroids, radii) arrays; tests that want a genuine shared-blob
conflict give the relevant indices the SAME position across controllers,
and tests that want no conflict give them distinct positions.

Uses stdlib unittest rather than pytest since pytest is not a declared
dependency of this project (not in requirements, not installed in .venv).
Run with:  python3 -m unittest tests.test_cold_conflict_resolution
"""
import unittest

import numpy as np

from src.controller import TrackingSystem
from src.transformations import Transform


def _T(x: float) -> Transform:
    """A world-frame pose at (x, 0, 0) -- only sol['T_world_ctrl'].t is ever
    read by these tests' own code paths (occlusion is never enabled here,
    so the physical-overlap check is the only T_world_ctrl consumer that
    actually runs). Callers space these >> min_controller_center_distance_m
    (default 0.05m) apart so this check stays inert and each test's own
    intended mechanism (shared-blob/chain/fixed-candidate) remains the
    deciding factor, exactly as before T_world_ctrl became required here."""
    return Transform(np.eye(3), np.array([x, 0.0, 0.0]))


def _make_tracking_system(matching_cfg=None, ctrl_trackers=None, cameras=None):
    """Build a bare TrackingSystem instance for testing _resolve_cold_conflicts
    in isolation, bypassing __init__ (which needs real cameras/models/pool
    wiring this method never touches when cross_controller_occlusion is off)."""
    ts = TrackingSystem.__new__(TrackingSystem)
    ts._matching_cfg = matching_cfg or {}
    ts.ctrl_trackers = ctrl_trackers or {}
    ts.cameras = cameras or {}
    return ts


def _geo(position_by_idx, radius=5.0):
    """Build a (centroids, radii) pair sized to cover every index in
    position_by_idx — the only indices any test ever reads. Unlisted
    indices below the max stay at (0, 0) and are never referenced."""
    max_idx = max(position_by_idx)
    centroids = np.zeros((max_idx + 1, 2), dtype=np.float64)
    radii = np.full(max_idx + 1, radius, dtype=np.float64)
    for idx, pos in position_by_idx.items():
        centroids[idx] = pos
    return centroids, radii


class ResolveColdConflictsTests(unittest.TestCase):
    def test_shared_blob_conflict_keeps_lower_error(self):
        # Both candidates' blob index 5 in camera 0 sit at the same real
        # pixel position — a genuine shared-blob conflict. ctrl_a has the
        # lower fused error and should be kept; ctrl_b should be dropped.
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0, "T_world_ctrl": _T(10.0),
                "assignment": [(5, 0), (6, 1)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 2.0, "T_world_ctrl": _T(20.0),
                "assignment": [(5, 0), (7, 1)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_a": {0: _geo({5: (500, 0), 6: (600, 0)})},
            "ctrl_b": {0: _geo({5: (500, 0), 7: (700, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons, _contested = ts._resolve_cold_conflicts(candidates, blob_geometry=blob_geometry)
        self.assertEqual(losers, {"ctrl_b"})

    def test_distinct_blob_positions_are_not_a_conflict(self):
        # Both candidates use blob index 5 in camera 0, but at genuinely
        # different pixel positions — two independent per-controller arrays
        # that just happen to share a local index. This must NOT be flagged
        # as a conflict (the bug this test guards against: raw index
        # equality across independently-detected arrays is meaningless).
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0, "T_world_ctrl": _T(30.0),
                "assignment": [(5, 0)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 2.0, "T_world_ctrl": _T(40.0),
                "assignment": [(5, 0)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_a": {0: _geo({5: (100, 0)})},
            "ctrl_b": {0: _geo({5: (900, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons, _contested = ts._resolve_cold_conflicts(candidates, blob_geometry=blob_geometry)
        self.assertEqual(losers, set())

    def test_missing_blob_geometry_skips_shared_blob_check(self):
        # No blob_geometry supplied at all -- shared-blob comparison must be
        # skipped entirely (never flagged) rather than falling back to raw
        # index equality.
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0, "T_world_ctrl": _T(50.0),
                "assignment": [(5, 0)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 2.0, "T_world_ctrl": _T(60.0),
                "assignment": [(5, 0)], "aux_assignments": {},
            },
        }
        ts = _make_tracking_system()
        losers, _reasons, _contested = ts._resolve_cold_conflicts(candidates)
        self.assertEqual(losers, set())

    def test_shared_blob_conflict_prefers_many_inliers_over_lowest_raw_error(self):
        # ctrl_a is a near-minimal single-camera fit sitting right at
        # min_inliers (2 pairs here, floor set to 2 below) with the lowest
        # raw error -- exactly the "lucky, barely-constrained fit" case that
        # used to win outright. ctrl_b has 5x the inlier evidence spread
        # across two cameras (primary + aux) with higher raw error. The
        # inlier-discounted score (error * min_inliers / total_pairs) should
        # now prefer ctrl_b: ctrl_a -> 0.20 * 2/2 = 0.20, ctrl_b -> 0.35 * 2/10
        # = 0.07. Blob index 5 in cam0 is a genuine shared position, so the
        # two do conflict and the score decides the winner.
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 0.20, "T_world_ctrl": _T(70.0),
                "assignment": [(5, 0), (6, 1)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 0.35, "T_world_ctrl": _T(80.0),
                "assignment": [(5, 10), (7, 11), (8, 12), (9, 13)],
                "aux_assignments": {1: [(0, 20), (1, 21), (2, 22), (3, 23), (4, 24), (6, 25)]},
            },
        }
        blob_geometry = {
            "ctrl_a": {0: _geo({5: (500, 0), 6: (600, 0)})},
            "ctrl_b": {0: _geo({5: (500, 0), 7: (700, 0), 8: (800, 0), 9: (900, 0)})},
        }
        ts = _make_tracking_system(matching_cfg={"min_inliers": 2})
        losers, _reasons, _contested = ts._resolve_cold_conflicts(candidates, blob_geometry=blob_geometry)
        self.assertEqual(losers, {"ctrl_a"})

    def test_three_node_chain_lets_far_end_win_once_middle_is_dropped(self):
        # Chain: ctrl_a -- ctrl_b (share blob position at cam0 index 1),
        # ctrl_b -- ctrl_c (share blob position at cam0 index 2). ctrl_a and
        # ctrl_c use different positions, so they are NOT in direct conflict.
        # ctrl_b has the highest error of the three, so the greedy resolution
        # should pick ctrl_a as winner first (lowest error, drops its only
        # neighbor ctrl_b), then pick ctrl_c as winner in a second round (its
        # only conflict, ctrl_b, is already gone) -- exercising the
        # "connected only through an already-dropped neighbor is free to win
        # later" case, not just pairwise comparison.
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0, "T_world_ctrl": _T(90.0),
                "assignment": [(1, 10)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 3.0, "T_world_ctrl": _T(100.0),
                "assignment": [(1, 20), (2, 21)], "aux_assignments": {},
            },
            "ctrl_c": {
                "primary_cam": 0, "error": 2.0, "T_world_ctrl": _T(110.0),
                "assignment": [(2, 30)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_a": {0: _geo({1: (100, 0)})},
            "ctrl_b": {0: _geo({1: (100, 0), 2: (200, 0)})},
            "ctrl_c": {0: _geo({2: (200, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons, _contested = ts._resolve_cold_conflicts(candidates, blob_geometry=blob_geometry)
        self.assertEqual(losers, {"ctrl_b"})

    def test_occlusion_disabled_skips_occlusion_branch_entirely(self):
        # Two candidates on different cameras (no possible shared-blob
        # overlap) with cross_controller_occlusion left at its default
        # (False). ctrl_trackers/cameras are deliberately left empty — if
        # the occlusion branch were entered despite the flag being off, it
        # would immediately KeyError on self.ctrl_trackers[occluder], so a
        # clean "no conflict" result here proves the branch was never
        # reached, not just that it happened to find no occlusion.
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0, "T_world_ctrl": _T(120.0),
                "assignment": [(1, 10)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 1, "error": 2.0, "T_world_ctrl": _T(130.0),
                "assignment": [(1, 20)], "aux_assignments": {},
            },
        }
        ts = _make_tracking_system(matching_cfg={"cross_controller_occlusion": False})
        losers, _reasons, _contested = ts._resolve_cold_conflicts(candidates)
        self.assertEqual(losers, set())

    def test_single_candidate_never_conflicts(self):
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0, "T_world_ctrl": _T(140.0),
                "assignment": [(1, 10)], "aux_assignments": {},
            },
        }
        ts = _make_tracking_system()
        losers, _reasons, _contested = ts._resolve_cold_conflicts(candidates)
        self.assertEqual(losers, set())

    def test_fixed_candidate_always_wins_even_with_higher_error(self):
        # ctrl_warm is passed as fixed (already committed this frame — the
        # cold-warm case): it shares blob position at cam0 index 1 with
        # ctrl_cold, and has a HIGHER error, so the plain error-based greedy
        # pass would normally keep ctrl_cold and drop ctrl_warm — but a fixed
        # candidate can never be dropped (its tracker state is already
        # committed), so ctrl_cold must lose regardless of the error
        # comparison.
        candidates = {
            "ctrl_cold": {
                "primary_cam": 0, "error": 0.5, "T_world_ctrl": _T(150.0),
                "assignment": [(1, 10)], "aux_assignments": {},
            },
            "ctrl_warm": {
                "primary_cam": 0, "error": 5.0, "T_world_ctrl": _T(160.0),
                "assignment": [(1, 20)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_cold": {0: _geo({1: (100, 0)})},
            "ctrl_warm": {0: _geo({1: (100, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons, _contested = ts._resolve_cold_conflicts(
            candidates, fixed_names={"ctrl_warm"}, blob_geometry=blob_geometry,
        )
        self.assertEqual(losers, {"ctrl_cold"})

    def test_two_fixed_candidates_conflicting_are_never_marked_losers(self):
        # Two fixed candidates sharing a blob position — a scenario this
        # method has no authority to resolve (neither can be un-committed) —
        # must leave both untouched rather than corrupting the loser set. A
        # third, non-fixed candidate conflicting with one of the fixed ones
        # must still be dropped normally.
        candidates = {
            "ctrl_warm_a": {
                "primary_cam": 0, "error": 1.0, "T_world_ctrl": _T(170.0),
                "assignment": [(1, 10)], "aux_assignments": {},
            },
            "ctrl_warm_b": {
                "primary_cam": 0, "error": 2.0, "T_world_ctrl": _T(180.0),
                "assignment": [(1, 20)], "aux_assignments": {},
            },
            "ctrl_cold": {
                "primary_cam": 0, "error": 0.1, "T_world_ctrl": _T(190.0),
                "assignment": [(1, 30)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_warm_a": {0: _geo({1: (100, 0)})},
            "ctrl_warm_b": {0: _geo({1: (100, 0)})},
            "ctrl_cold":   {0: _geo({1: (100, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons, _contested = ts._resolve_cold_conflicts(
            candidates, fixed_names={"ctrl_warm_a", "ctrl_warm_b"}, blob_geometry=blob_geometry,
        )
        self.assertEqual(losers, {"ctrl_cold"})

    def test_no_fixed_names_matches_prior_behavior(self):
        # fixed_names omitted entirely — must behave exactly like the
        # original cold-cold-only signature (lower error wins).
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0, "T_world_ctrl": _T(200.0),
                "assignment": [(5, 0)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 2.0, "T_world_ctrl": _T(210.0),
                "assignment": [(5, 0)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_a": {0: _geo({5: (500, 0)})},
            "ctrl_b": {0: _geo({5: (500, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons, _contested = ts._resolve_cold_conflicts(candidates, blob_geometry=blob_geometry)
        self.assertEqual(losers, {"ctrl_b"})


class ContestedWinnersTests(unittest.TestCase):
    """Regression for _resolve_cold_conflicts' third return value,
    contested_winners (2026-09-13) -- ctrl_names that WON (survived, not in
    losers) despite being part of at least one real conflict this frame.
    Added after a real case: a 6-inlier/0.32px bootstrap won a shared-blob
    conflict outright (an otherwise perfectly ordinary-looking confidence/
    error/inlier profile) and turned out 132.7deg/1.63m wrong vs mocap
    ground truth -- "won" only means "beat whichever candidate it directly
    conflicted with," not "is correct." See _commit_fused_solution's own
    winner_was_contested parameter for how callers use this."""

    def test_uncontested_winner_is_not_in_contested_set(self):
        # No conflict at all (distinct blob positions) -- neither survivor
        # should appear in contested_winners.
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0, "T_world_ctrl": _T(1000.0),
                "assignment": [(5, 0)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 2.0, "T_world_ctrl": _T(1010.0),
                "assignment": [(5, 0)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_a": {0: _geo({5: (100, 0)})},
            "ctrl_b": {0: _geo({5: (900, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons, contested = ts._resolve_cold_conflicts(candidates, blob_geometry=blob_geometry)
        self.assertEqual(losers, set())
        self.assertEqual(contested, set())

    def test_winner_of_a_real_conflict_is_contested(self):
        # Same shared-blob setup as test_shared_blob_conflict_keeps_lower_error:
        # ctrl_a wins outright, but it WAS part of a real conflict -- must be
        # flagged contested even though it's not a loser.
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0, "T_world_ctrl": _T(1020.0),
                "assignment": [(5, 0), (6, 1)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 2.0, "T_world_ctrl": _T(1030.0),
                "assignment": [(5, 0), (7, 1)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_a": {0: _geo({5: (500, 0), 6: (600, 0)})},
            "ctrl_b": {0: _geo({5: (500, 0), 7: (700, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons, contested = ts._resolve_cold_conflicts(candidates, blob_geometry=blob_geometry)
        self.assertEqual(losers, {"ctrl_b"})
        self.assertEqual(contested, {"ctrl_a"})

    def test_three_node_chain_only_flags_the_nodes_that_actually_conflicted(self):
        # Same chain as test_three_node_chain_lets_far_end_win_once_middle_is_
        # dropped: ctrl_a wins round 1 (conflicted with ctrl_b -> contested),
        # ctrl_c wins round 2 with NO remaining conflicts (its only conflict,
        # ctrl_b, was already dropped) -- but ctrl_c DID conflict with ctrl_b
        # originally, so it's contested too; conflicts[] is fixed at graph-
        # build time, not reduced by drops.
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0, "T_world_ctrl": _T(1040.0),
                "assignment": [(1, 10)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 3.0, "T_world_ctrl": _T(1050.0),
                "assignment": [(1, 20), (2, 21)], "aux_assignments": {},
            },
            "ctrl_c": {
                "primary_cam": 0, "error": 2.0, "T_world_ctrl": _T(1060.0),
                "assignment": [(2, 30)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_a": {0: _geo({1: (100, 0)})},
            "ctrl_b": {0: _geo({1: (100, 0), 2: (200, 0)})},
            "ctrl_c": {0: _geo({2: (200, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons, contested = ts._resolve_cold_conflicts(candidates, blob_geometry=blob_geometry)
        self.assertEqual(losers, {"ctrl_b"})
        self.assertEqual(contested, {"ctrl_a", "ctrl_c"})

    def test_fixed_winner_that_conflicted_is_contested(self):
        # Same setup as test_fixed_candidate_always_wins_even_with_higher_error:
        # ctrl_warm is fixed and wins by fiat, but it DID conflict with
        # ctrl_cold -- must still be flagged contested (fixed winners aren't
        # exempt from this signal, only from being droppable).
        candidates = {
            "ctrl_cold": {
                "primary_cam": 0, "error": 0.5, "T_world_ctrl": _T(1070.0),
                "assignment": [(1, 10)], "aux_assignments": {},
            },
            "ctrl_warm": {
                "primary_cam": 0, "error": 5.0, "T_world_ctrl": _T(1080.0),
                "assignment": [(1, 20)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_cold": {0: _geo({1: (100, 0)})},
            "ctrl_warm": {0: _geo({1: (100, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons, contested = ts._resolve_cold_conflicts(
            candidates, fixed_names={"ctrl_warm"}, blob_geometry=blob_geometry,
        )
        self.assertEqual(losers, {"ctrl_cold"})
        self.assertEqual(contested, {"ctrl_warm"})


if __name__ == "__main__":
    unittest.main()
