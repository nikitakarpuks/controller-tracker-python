"""Regression coverage for src/controller.py's _weak_solo_accept_cids (added
2026-09-11), shared by ControllerTracker.update and TrackingSystem.update_
warm_batch.

Bug: both call sites used to accept a controller's cheap-search result the
instant ANY camera produced something -- "Only if NO camera produced
anything this frame do we pay for brute-force recovery" (see
ControllerTracker.update's own docstring) -- even when that one camera's
match was weak (few inliers, or several of its own detected blobs left
unmatched) and ANOTHER camera that had real detected blobs this frame came
up with nothing. Confirmed on a real recording (frame_range 800-900
relative frames 50/51/53): cam1 kept accepting 4-7-inlier proximity matches
while cam0 -- which had its own real candidate blobs every one of those
frames -- never got to independently confirm or contradict it, letting one
weak, partially-wrong camera lock in the whole controller's fused pose frame
after frame with no cross-check.

Fix: _weak_solo_accept_cids flags this pattern -- every accepted solution is
"weak" (inliers < strong_match_inliers OR inliers/available-blobs ratio <
weak_solo_blob_utilization_floor) AND at least one other camera that had
blobs this frame contributed nothing -- so the caller can force a
brute-force cross-check (ControllerTracker.update, when allow_brute) or
defer to the existing cheap-search-failure fallback chain
(TrackingSystem.update_warm_batch, which has no brute-force of its own).

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_weak_solo_accept
"""
import unittest

from src.controller import _weak_solo_accept_cids

_CFG = {"strong_match_inliers": 6, "weak_solo_blob_utilization_floor": 0.7}


def _cs(cam_id: int, n_inliers: int) -> dict:
    return {"cam_id": cam_id, "solution": {"assignment": [(i, i) for i in range(n_inliers)]}}


class WeakSoloAcceptTests(unittest.TestCase):
    def test_real_frame_50_shape_low_ratio_high_inlier_count(self):
        """Real case: cam1 accepted 7 inliers (>= strong_match_inliers=6 on
        its own) but only matched 9 of its own 13 detected blobs -- 7/13 =
        0.538 is well under the 0.7 utilization floor. cam0 had blobs but no
        accepted solution. Inlier count alone (the first thing tried) misses
        this; the ratio check is load-bearing here."""
        cam_solutions = [_cs(cam_id=1, n_inliers=7)]
        result = _weak_solo_accept_cids(
            cam_solutions, eligible_cids=[0, 1], av_count_by_cid={1: 13}, matching_cfg=_CFG)
        self.assertEqual(result, [0])

    def test_real_frame_51_shape_low_inlier_count(self):
        """Real case: cam1 accepted only 4 inliers (< strong_match_inliers=6)
        while cam0 had blobs but no accepted solution."""
        cam_solutions = [_cs(cam_id=1, n_inliers=4)]
        result = _weak_solo_accept_cids(
            cam_solutions, eligible_cids=[0, 1], av_count_by_cid={1: 6}, matching_cfg=_CFG)
        self.assertEqual(result, [0])

    def test_strong_accept_not_flagged_even_with_other_camera_failing(self):
        """A genuinely strong, high-utilization single-camera accept (e.g.
        the other camera is legitimately out of view this frame) must NOT
        be second-guessed -- forcing brute-force on every such frame would
        be a real, needless performance regression."""
        cam_solutions = [_cs(cam_id=1, n_inliers=8)]
        result = _weak_solo_accept_cids(
            cam_solutions, eligible_cids=[0, 1], av_count_by_cid={1: 8}, matching_cfg=_CFG)
        self.assertEqual(result, [])

    def test_no_other_eligible_camera_never_flagged(self):
        """Only one camera had blobs at all this frame (the other is out of
        frustum / had zero detections, never even in eligible_cids) -- there
        is nothing to cross-check against, regardless of how weak the sole
        accept is."""
        cam_solutions = [_cs(cam_id=1, n_inliers=2)]
        result = _weak_solo_accept_cids(
            cam_solutions, eligible_cids=[1], av_count_by_cid={1: 20}, matching_cfg=_CFG)
        self.assertEqual(result, [])

    def test_no_accepted_solutions_never_flagged(self):
        """Nothing accepted at all -- this is the pre-existing 'not
        cam_solutions' case, a different code path entirely; must not also
        be reported here."""
        result = _weak_solo_accept_cids(
            [], eligible_cids=[0, 1], av_count_by_cid={}, matching_cfg=_CFG)
        self.assertEqual(result, [])

    def test_every_eligible_camera_already_solved_never_flagged(self):
        """Both eligible cameras already contributed (a real, cross-validated
        multi-camera accept) -- nothing left to check, regardless of how weak
        either individual camera's own fit looks."""
        cam_solutions = [_cs(cam_id=0, n_inliers=4), _cs(cam_id=1, n_inliers=4)]
        result = _weak_solo_accept_cids(
            cam_solutions, eligible_cids=[0, 1], av_count_by_cid={0: 10, 1: 10}, matching_cfg=_CFG)
        self.assertEqual(result, [])

    def test_one_strong_one_weak_of_multiple_accepts_not_flagged(self):
        """Only flags when EVERY accepted solution is weak -- if at least one
        accepted camera is already strong, that's real corroboration and the
        other failing camera doesn't need a forced cross-check."""
        cam_solutions = [_cs(cam_id=0, n_inliers=10)]
        result = _weak_solo_accept_cids(
            cam_solutions, eligible_cids=[0, 1, 2], av_count_by_cid={0: 10}, matching_cfg=_CFG)
        self.assertEqual(result, [])

    def test_multiple_failed_cameras_all_returned(self):
        cam_solutions = [_cs(cam_id=1, n_inliers=3)]
        result = _weak_solo_accept_cids(
            cam_solutions, eligible_cids=[0, 1, 2], av_count_by_cid={1: 8}, matching_cfg=_CFG)
        self.assertEqual(sorted(result), [0, 2])


if __name__ == "__main__":
    unittest.main()
