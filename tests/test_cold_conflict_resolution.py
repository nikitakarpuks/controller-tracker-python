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
                "primary_cam": 0, "error": 1.0,
                "assignment": [(5, 0), (6, 1)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 2.0,
                "assignment": [(5, 0), (7, 1)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_a": {0: _geo({5: (500, 0), 6: (600, 0)})},
            "ctrl_b": {0: _geo({5: (500, 0), 7: (700, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons = ts._resolve_cold_conflicts(candidates, blob_geometry=blob_geometry)
        self.assertEqual(losers, {"ctrl_b"})

    def test_distinct_blob_positions_are_not_a_conflict(self):
        # Both candidates use blob index 5 in camera 0, but at genuinely
        # different pixel positions — two independent per-controller arrays
        # that just happen to share a local index. This must NOT be flagged
        # as a conflict (the bug this test guards against: raw index
        # equality across independently-detected arrays is meaningless).
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0,
                "assignment": [(5, 0)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 2.0,
                "assignment": [(5, 0)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_a": {0: _geo({5: (100, 0)})},
            "ctrl_b": {0: _geo({5: (900, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons = ts._resolve_cold_conflicts(candidates, blob_geometry=blob_geometry)
        self.assertEqual(losers, set())

    def test_missing_blob_geometry_skips_shared_blob_check(self):
        # No blob_geometry supplied at all -- shared-blob comparison must be
        # skipped entirely (never flagged) rather than falling back to raw
        # index equality.
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0,
                "assignment": [(5, 0)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 2.0,
                "assignment": [(5, 0)], "aux_assignments": {},
            },
        }
        ts = _make_tracking_system()
        losers, _reasons = ts._resolve_cold_conflicts(candidates)
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
                "primary_cam": 0, "error": 0.20,
                "assignment": [(5, 0), (6, 1)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 0.35,
                "assignment": [(5, 10), (7, 11), (8, 12), (9, 13)],
                "aux_assignments": {1: [(0, 20), (1, 21), (2, 22), (3, 23), (4, 24), (6, 25)]},
            },
        }
        blob_geometry = {
            "ctrl_a": {0: _geo({5: (500, 0), 6: (600, 0)})},
            "ctrl_b": {0: _geo({5: (500, 0), 7: (700, 0), 8: (800, 0), 9: (900, 0)})},
        }
        ts = _make_tracking_system(matching_cfg={"min_inliers": 2})
        losers, _reasons = ts._resolve_cold_conflicts(candidates, blob_geometry=blob_geometry)
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
                "primary_cam": 0, "error": 1.0,
                "assignment": [(1, 10)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 3.0,
                "assignment": [(1, 20), (2, 21)], "aux_assignments": {},
            },
            "ctrl_c": {
                "primary_cam": 0, "error": 2.0,
                "assignment": [(2, 30)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_a": {0: _geo({1: (100, 0)})},
            "ctrl_b": {0: _geo({1: (100, 0), 2: (200, 0)})},
            "ctrl_c": {0: _geo({2: (200, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons = ts._resolve_cold_conflicts(candidates, blob_geometry=blob_geometry)
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
                "primary_cam": 0, "error": 1.0,
                "assignment": [(1, 10)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 1, "error": 2.0,
                "assignment": [(1, 20)], "aux_assignments": {},
            },
        }
        ts = _make_tracking_system(matching_cfg={"cross_controller_occlusion": False})
        losers, _reasons = ts._resolve_cold_conflicts(candidates)
        self.assertEqual(losers, set())

    def test_single_candidate_never_conflicts(self):
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0,
                "assignment": [(1, 10)], "aux_assignments": {},
            },
        }
        ts = _make_tracking_system()
        losers, _reasons = ts._resolve_cold_conflicts(candidates)
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
                "primary_cam": 0, "error": 0.5,
                "assignment": [(1, 10)], "aux_assignments": {},
            },
            "ctrl_warm": {
                "primary_cam": 0, "error": 5.0,
                "assignment": [(1, 20)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_cold": {0: _geo({1: (100, 0)})},
            "ctrl_warm": {0: _geo({1: (100, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons = ts._resolve_cold_conflicts(
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
                "primary_cam": 0, "error": 1.0,
                "assignment": [(1, 10)], "aux_assignments": {},
            },
            "ctrl_warm_b": {
                "primary_cam": 0, "error": 2.0,
                "assignment": [(1, 20)], "aux_assignments": {},
            },
            "ctrl_cold": {
                "primary_cam": 0, "error": 0.1,
                "assignment": [(1, 30)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_warm_a": {0: _geo({1: (100, 0)})},
            "ctrl_warm_b": {0: _geo({1: (100, 0)})},
            "ctrl_cold":   {0: _geo({1: (100, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons = ts._resolve_cold_conflicts(
            candidates, fixed_names={"ctrl_warm_a", "ctrl_warm_b"}, blob_geometry=blob_geometry,
        )
        self.assertEqual(losers, {"ctrl_cold"})

    def test_no_fixed_names_matches_prior_behavior(self):
        # fixed_names omitted entirely — must behave exactly like the
        # original cold-cold-only signature (lower error wins).
        candidates = {
            "ctrl_a": {
                "primary_cam": 0, "error": 1.0,
                "assignment": [(5, 0)], "aux_assignments": {},
            },
            "ctrl_b": {
                "primary_cam": 0, "error": 2.0,
                "assignment": [(5, 0)], "aux_assignments": {},
            },
        }
        blob_geometry = {
            "ctrl_a": {0: _geo({5: (500, 0)})},
            "ctrl_b": {0: _geo({5: (500, 0)})},
        }
        ts = _make_tracking_system()
        losers, _reasons = ts._resolve_cold_conflicts(candidates, blob_geometry=blob_geometry)
        self.assertEqual(losers, {"ctrl_b"})


if __name__ == "__main__":
    unittest.main()
