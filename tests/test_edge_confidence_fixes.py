"""Regression coverage for the 2026-09-13 fixes to the KB4 edge-confidence work
(src/pose_search.py): two bugs in the proximity-tracking reprojection-threshold
widening, plus the new coverage-only aux-threshold soft credit.

Bug 1 (ceiling too loose): _edge_widened_reproj_px's multiplier alone let
proximity_reprojection_threshold's real config value (2.5px) * edge_reproj_widen_max
(2.5x) reach 6.25px -- looser than every one of this project's own brute-force
thresholds (1.5-2.0px). Fixed via edge_reproj_widen_abs_cap_px, applied in the new
shared _edge_widened_thresh helper.

Bug 2 (max-over-locked-pairs): the widen decision must depend ONLY on the pairs
actually being newly tested this call (hyp_assignment), never on pairs already
CONFIRMED/locked from a prior frame -- a stale locked pair drifted near the edge
must not widen tolerance for an unrelated, brand-new, center-of-frame pair.
_edge_widened_reproj_px itself is a pure function of whatever new_blob_px it's
given, so this is tested as a contract at that boundary (see
EdgeWidenedReprojPxTests.test_contract_depends_only_on_passed_points); the call
sites in proximity_search were fixed to pass hyp_pairs, not locked+hyp.

Coverage-only soft credit: brute_search_tier's aux-camera loop now gives a near-miss
blob (fails the flat/unwidened brute_aux_reprojection_threshold_px hard gate, but
falls within the radially-widened soft tolerance) partial credit toward led_cov's
numerator ONLY -- never written into aux_assignments/aux_cameras/extra_inlier_count,
which feed cross-controller conflict resolution and tie-breaking in
src/controller.py (a false aux MATCH there is far more consequential than a
coverage nudge; critical review rejected widening the hard gate directly for this
reason). AuxSoftCreditCoverageTests exercises this against a real two-camera KB4
scenario (a facing-visible LED genuinely at 93% of the aux camera's own rpmax_px,
shifted 2.8px off its true projection -- comfortably between the flat 2.0px hard
gate and the ~3.5px soft/capped threshold at that radius).

Uses a self-contained synthetic kb4 camera (real cam0 intrinsics from this
project's calibration, hardcoded) rather than loading data/cameras/kb4_calib.json
-- deliberately, so these tests run regardless of that file's presence (see
tests/test_camera_kb4_rpmax.py's own note on that pre-existing environment gap).

Uses stdlib unittest rather than pytest (see tests/test_cold_conflict_resolution.py).
Run with:  python3 -m unittest tests.test_edge_confidence_fixes
"""
import json
import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.camera import Camera
from src.controller import ControllerModel, create_leds_from_config
from src.pose_search import PoseSearcher
from src.transformations import Transform

REPO_ROOT = Path(__file__).resolve().parent.parent
CTRL_PATH = REPO_ROOT / "data" / "controllers" / "right_controller_A85K6081930636R.json"

# Real cam0 kb4 intrinsics from this project's live calibration, hardcoded so
# these tests don't depend on data/cameras/kb4_calib.json being present.
_KB4_INTR = {
    "camera_type": "kb4",
    "intrinsics": {
        "fx": 269.5212037284437, "fy": 269.233878887661,
        "cx": 326.30986878388745, "cy": 235.25250025580684,
        "k1": 0.0804888292241004, "k2": -0.03996716237452026,
        "k3": 0.2230177662647582, "k4": -0.12633072574931184,
    },
}
_RES = [640, 480]


def _make_camera(camera_idx=0, extrinsics=None):
    t_a = {"px": 0.0, "py": 0.0, "pz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    entries = [t_a] if extrinsics is None else [t_a, extrinsics]
    cfg = {"value0": {"T_imu_cam": entries,
                       "intrinsics": [_KB4_INTR] * len(entries),
                       "resolution": [_RES] * len(entries)}}
    return Camera(cfg, camera_idx=camera_idx)


def _make_pose_searcher(cam, matching_cfg=None):
    with open(CTRL_PATH) as f:
        ctrl_cfg = json.load(f)
    model = ControllerModel(create_leds_from_config(ctrl_cfg), "right")
    return PoseSearcher(cam, model, geometry_cfg={}, matching_cfg=matching_cfg or {})


_MATCHING_CFG = {
    "proximity_reprojection_threshold": 2.5,   # real live config value, not the 2.0 code default
    "brute_aux_reprojection_threshold_px": 2.0,
    "edge_confidence_inner_fraction": 0.8,
    "edge_confidence_floor": 0.3,
    "edge_reproj_widen_max": 2.5,
    "edge_reproj_widen_abs_cap_px": 3.5,
}


class EdgeWidenedThreshTests(unittest.TestCase):
    """_edge_widened_thresh (src/pose_search.py) -- the shared widen+cap helper."""

    def setUp(self):
        self.cam = _make_camera()
        self.ps = _make_pose_searcher(self.cam, _MATCHING_CFG)

    def test_no_widen_within_inner_fraction(self):
        r_px = self.cam.rpmax_px * 0.5  # well inside inner_fraction=0.8
        got = self.ps._edge_widened_thresh(2.5, r_px, self.cam.rpmax_px)
        self.assertAlmostEqual(got, 2.5, places=6)

    def test_bug1_regression_capped_below_naive_multiplier_ceiling(self):
        """2.5px (real proximity_reprojection_threshold) * 2.5 (edge_reproj_widen_max)
        = 6.25px naively -- looser than every brute-force threshold in this
        project (1.5-2.0px). Must be capped at edge_reproj_widen_abs_cap_px (3.5)
        instead. Fails without the fix (would assert 6.25 <= 3.5, False)."""
        r_px = self.cam.rpmax_px  # right at the boundary -> max widen
        got = self.ps._edge_widened_thresh(2.5, r_px, self.cam.rpmax_px)
        self.assertLessEqual(got, 3.5 + 1e-9)
        self.assertAlmostEqual(got, 3.5, places=6)

    def test_never_narrower_than_base_even_with_misconfigured_cap(self):
        ps = _make_pose_searcher(self.cam, {**_MATCHING_CFG, "edge_reproj_widen_abs_cap_px": 1.0})
        got = ps._edge_widened_thresh(2.5, self.cam.rpmax_px, self.cam.rpmax_px)
        self.assertGreaterEqual(got, 2.5)

    def test_vectorised_array_input_capped_elementwise(self):
        r_px = np.array([0.0, self.cam.rpmax_px * 0.5, self.cam.rpmax_px])
        got = self.ps._edge_widened_thresh(2.5, r_px, self.cam.rpmax_px)
        self.assertIsInstance(got, np.ndarray)
        np.testing.assert_allclose(got, [2.5, 2.5, 3.5], atol=1e-6)


class EdgeWidenedReprojPxTests(unittest.TestCase):
    """_edge_widened_reproj_px -- proximity_search's own call-site wrapper."""

    def setUp(self):
        self.cam = _make_camera()
        self.ps = _make_pose_searcher(self.cam, _MATCHING_CFG)

    def test_empty_input_returns_flat_base(self):
        got = self.ps._edge_widened_reproj_px(np.zeros((0, 2), dtype=np.float32))
        self.assertEqual(got, 2.5)

    def test_center_of_frame_blob_no_widen(self):
        blob = np.array([[self.cam.cx, self.cam.cy]], dtype=np.float32)
        got = self.ps._edge_widened_reproj_px(blob)
        self.assertAlmostEqual(got, 2.5, places=3)

    def test_edge_blob_widens_and_is_capped(self):
        blob = np.array([[self.cam.cx + self.cam.rpmax_px, self.cam.cy]], dtype=np.float32)
        got = self.ps._edge_widened_reproj_px(blob)
        self.assertGreater(got, 2.5)
        self.assertLessEqual(got, 3.5 + 1e-6)

    def test_bug2_regression_contract_depends_only_on_passed_points(self):
        """The fix for bug 2 lives in proximity_search's call sites (passing
        hyp_pairs, not locked_assignment + hyp_assignment) -- this pins the
        contract _edge_widened_reproj_px must honour for that fix to matter:
        a locked pair sitting at the boundary must NOT influence the result
        when it isn't included in new_blob_px. Simulates "correct" (hyp-only,
        center blob) vs what the pre-fix code would have passed (hyp + a
        stale locked pair at the edge)."""
        center_blob = np.array([[self.cam.cx, self.cam.cy]], dtype=np.float32)
        hyp_only = self.ps._edge_widened_reproj_px(center_blob)
        self.assertAlmostEqual(hyp_only, 2.5, places=3)  # correct (fixed) behaviour

        locked_near_edge = np.array([[self.cam.cx + self.cam.rpmax_px, self.cam.cy]], dtype=np.float32)
        locked_plus_hyp = np.array(
            np.concatenate([center_blob, locked_near_edge], axis=0), dtype=np.float32,
        )
        pre_fix_behaviour = self.ps._edge_widened_reproj_px(locked_plus_hyp)
        self.assertGreater(pre_fix_behaviour, hyp_only)  # the bug this fix prevents


# ---------------------------------------------------------------------------
# Coverage-only aux soft credit -- real two-camera brute_search_tier scenario.
# ---------------------------------------------------------------------------

# Camera B extrinsic (offset + yaw) chosen so several real, facing-visible LEDs
# of the right controller (ground-truth pose: 90deg about Y, 0.35m along camera
# A's axis) land at 81-93% of camera B's own rpmax_px -- found by sweeping
# offset/yaw until PoseSearcher._aux_cam_vis's own (production) visibility
# check, not a hand-rolled approximation, put several LEDs there.
_CAM_B_YAW_DEG = -80.0
_CAM_B_OFFSET = {"px": 0.15, "py": 0.2, "pz": 0.0}
_TARGET_LED = 27  # lands at ~93% of cam B's rpmax_px in this configuration
_NEAR_MISS_SHIFT_PX = 2.8  # between the flat 2.0px hard gate and the ~3.5px capped soft threshold there


def _make_aux_soft_credit_fixture():
    cam_a = _make_camera(camera_idx=0)
    qb = R.from_euler("y", _CAM_B_YAW_DEG, degrees=True).as_quat()
    extrinsic_b = {**_CAM_B_OFFSET,
                   "qx": float(qb[0]), "qy": float(qb[1]), "qz": float(qb[2]), "qw": float(qb[3])}
    cam_b = _make_camera(camera_idx=1, extrinsics=extrinsic_b)

    with open(CTRL_PATH) as f:
        ctrl_cfg = json.load(f)
    model = ControllerModel(create_leds_from_config(ctrl_cfg), "right")

    R_ctrl = R.from_euler("xy", [0, 90], degrees=True).as_matrix().astype(np.float32)
    t_ctrl = np.array([0.0, 0.0, 0.35], dtype=np.float32)
    T_world_ctrl = Transform(R_ctrl, t_ctrl)

    vis_ids_a = np.array([2, 7, 9, 11, 13])
    rvec_a, _ = cv2.Rodrigues(R_ctrl)
    blobs_a, _ = cam_a.project_points(model.positions[vis_ids_a], rvec_a, t_ctrl)
    blobs_a = blobs_a.astype(np.float32)

    # Use PoseSearcher's own _aux_cam_vis (the actual production visibility
    # check, not a hand-rolled approximation) to get camera B's real visible
    # LED set and their true projected pixel positions.
    probe = PoseSearcher(cam_a, model, geometry_cfg={}, matching_cfg={})
    R_i, t_i, rv_i, _, vis_i = probe._aux_cam_vis(cam_b, T_world_ctrl, None)
    vis_ids_b = np.where(vis_i)[0]
    blobs_b_true, _ = cam_b.project_points(model.positions[vis_ids_b], rv_i, t_i)
    blobs_b_true = blobs_b_true.astype(np.float32)

    target_idx = int(np.where(vis_ids_b == _TARGET_LED)[0][0])
    r_target = float(np.hypot(blobs_b_true[target_idx, 0] - cam_b.cx,
                               blobs_b_true[target_idx, 1] - cam_b.cy))
    assert 0.85 < r_target / cam_b.rpmax_px < 1.0, \
        "fixture assumption broken: target LED no longer near cam B's rpmax_px boundary"

    return cam_a, cam_b, model, blobs_a, blobs_b_true, target_idx


def _run_brute(cam_a, model, blobs_a, other_cameras_blobs, matching_cfg):
    ps = PoseSearcher(cam_a, model, geometry_cfg={}, matching_cfg=matching_cfg)
    mask = np.ones(len(blobs_a), dtype=bool)
    state = ps.new_brute_state(blobs_a, pose_prior=None, other_cameras_blobs=other_cameras_blobs,
                                blob_mask=mask, occluders_per_cam=None)
    for tier_idx in range(len(ps._c_brute_depth_tiers)):
        ps.brute_search_tier(state, tier_idx)
        if state.strong_found:
            break
    return ps.finalize_brute_state(state)


class AuxSoftCreditCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.cam_a, cls.cam_b, cls.model, cls.blobs_a,
         cls.blobs_b_true, cls.target_idx) = _make_aux_soft_credit_fixture()

    def _run(self, edge_reproj_widen_max, shift_px):
        blobs_b = self.blobs_b_true.copy()
        blobs_b[self.target_idx, 0] += shift_px
        cfg = {**_MATCHING_CFG, "edge_reproj_widen_max": edge_reproj_widen_max}
        return _run_brute(self.cam_a, self.model, self.blobs_a,
                           [(self.cam_b, blobs_b, None)], cfg)

    def test_exact_match_unaffected_by_widen_setting(self):
        sol_on = self._run(2.5, 0.0)
        sol_off = self._run(1.0, 0.0)
        self.assertEqual(sol_on["aux_inliers"], sol_off["aux_inliers"])
        self.assertEqual(sol_on["aux_assignments"], sol_off["aux_assignments"])
        self.assertAlmostEqual(sol_on["led_cov"], sol_off["led_cov"], places=6)

    def test_soft_credit_improves_coverage_without_touching_hard_match(self):
        """Core regression for the coverage-only design: a near-miss blob (2.8px
        off -- fails the flat 2.0px hard gate, within the ~3.5px capped soft
        threshold at this LED's radius) must raise led_cov when the soft credit
        is active, while aux_inliers/aux_assignments (the hard-match signals that
        feed cross-controller conflict resolution in controller.py) stay
        byte-for-byte identical either way. Fails without the soft-credit fix
        (both configs would produce identical led_cov)."""
        sol_on = self._run(2.5, _NEAR_MISS_SHIFT_PX)
        sol_off = self._run(1.0, _NEAR_MISS_SHIFT_PX)
        self.assertIsNotNone(sol_on)
        self.assertIsNotNone(sol_off)

        # Hard-match signals: untouched by the soft credit.
        self.assertEqual(sol_on["aux_inliers"], sol_off["aux_inliers"])
        self.assertEqual(sol_on["aux_assignments"], sol_off["aux_assignments"])
        self.assertNotIn(self._target_pair(), sol_on["aux_assignments"].get(1, []))

        # Coverage-only signal: measurably higher with the soft credit active.
        self.assertGreater(sol_on["led_cov"], sol_off["led_cov"])

    def test_soft_credit_itself_capped_beyond_soft_threshold(self):
        """A near-miss far enough that even the widened/capped soft threshold
        doesn't reach it (per bug 1's own cap) must NOT get any credit --
        led_cov should match the no-credit baseline."""
        far_shift = 6.0
        sol_on = self._run(2.5, far_shift)
        sol_off = self._run(1.0, far_shift)
        self.assertIsNotNone(sol_on)
        self.assertIsNotNone(sol_off)
        self.assertAlmostEqual(sol_on["led_cov"], sol_off["led_cov"], places=6)

    def _target_pair(self):
        # blobs_b's row order == vis_ids_b order 1:1 (no reordering in this
        # fixture), so the row index in aux_assignments is target_idx itself.
        return (self.target_idx, _TARGET_LED)


if __name__ == "__main__":
    unittest.main()
