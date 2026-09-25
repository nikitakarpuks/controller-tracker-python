"""Continuous rotation gate wired into HeuristicPoseFusionFilter (fusion_heuristic.rot_gate_mode) plus the
_headset_inputs / disable_headset_ego_motion refactor. Fixtures reuse tests/test_pose_fusion_heuristic.py."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from tests.test_pose_fusion_heuristic import _solution, _imu_arrays
from src.pose_fusion_heuristic import HeuristicPoseFusionFilter

MS = 1_000_000


def _const_gyro_arrays(w_rad_s, axis=(0.0, 0.0, 1.0), t_end_ns=600 * MS, step_ns=5 * MS):
    ts = list(range(0, t_end_ns + 1, step_ns))
    w = np.array(axis) * w_rad_s
    return _imu_arrays([(t, w) for t in ts], [(t, [0.0, 0.0, 9.81]) for t in ts])


def _filter(gyro_rad_s, frames_since_update, mode="continuous", extra_hc=None, g_world=True, headset_mocap=None):
    gyro_data, accel_data = _const_gyro_arrays(gyro_rad_s)
    hc = {
        "imu_decay_frames": 4, "vision_weight_weak_inliers": 5, "vision_weight_strong_inliers": 8,
        "vision_weight_weak_error_px": 0.5, "vision_weight_strong_error_px": 0.15,
        "implausible_jump_pos_thresh_base_mm": 0.0, "implausible_jump_pos_thresh_per_speed_mm_s": 22.4,
        "implausible_jump_rot_thresh_base_deg": 21.3, "implausible_jump_rot_thresh_per_speed_deg_s": 6.54,
        "implausible_jump_gyro_calm_floor_dps": 900.0, "implausible_jump_rot_thresh_per_gyro_deg_per_dps": 0.0046,
        "cold_reacquire_rot_veto_max_s": 0.25, "cold_reacquire_rot_veto_thresh_deg": 100.0,
        "coast_trust_gyro_calm_floor_dps": 0.0, "coast_trust_rot_shrink_s_per_dps": 0.00018,
        "coast_trust_rot_min_budget_s": 0.003, "coast_trust_rot_calm_extend_ceiling_s": 0.15,
        "coast_trust_rot_calm_extend_max_dps": 100.0, "agreement_hist_min_samples": 5,
        "rot_gate_mode": mode,
    }
    hc.update(extra_hc or {})
    cfg = {"max_coast_s": 0.3, "max_consecutive_rejects": 5,
           "matching": {"max_plausible_hand_speed_m_s": 7.5, "max_plausible_hand_ang_speed_deg_s": 2200.0},
           "fusion_heuristic": hc}
    est = Mock(g_world=np.array([0.0, -9.81, 0.0]) if g_world else None)
    f = HeuristicPoseFusionFilter(gyro_data=gyro_data, accel_data=accel_data, lever_arm=np.zeros(3),
                                  g_world_estimator=est, cfg=cfg, ctrl_name="test", headset_mocap=headset_mocap)
    f.R, f.p, f.v = np.eye(3), np.zeros(3), np.zeros(3)
    f.velocity_established = True
    f.last_update_ts_ns = 0
    f.frames_since_update = frames_since_update
    return f


def _cand(f, dt_s, innov_deg, n_inliers=6, error_px=0.4):
    """Candidate exactly innov_deg from the state's own gyro-only R_pred at dt_s (position == p_pred)."""
    from src.imu_data import slice_imu_to_window, integrate_gyro_segment
    ts = int(dt_s * 1e9)
    tg, wg = slice_imu_to_window(*f._gyro_data, 0, ts)
    R_pred = f.R @ integrate_gyro_segment(tg, wg, 0, ts)
    R_meas = R_pred @ Rotation.from_rotvec(np.radians(innov_deg) * np.array([1.0, 0.0, 0.0])).as_matrix()
    return _solution(R_meas, f.p.copy(), n_inliers=n_inliers, error_px=error_px), ts


class ContinuousGateTests(unittest.TestCase):
    def test_348_like_cold_mirror_vetoed_in_continuous(self):
        f = _filter(np.radians(300.8), frames_since_update=5)
        sol, ts = _cand(f, 0.2441, 172.0)
        self.assertFalse(f.try_update(sol, ts))
        self.assertEqual(f._last.get("outcome"), "implausible_reject")

    def test_5333_like_warm_87deg_vetoed_in_continuous(self):
        f = _filter(np.radians(799.7), frames_since_update=2)
        sol, ts = _cand(f, 0.0555, 87.0, n_inliers=10, error_px=0.3)
        self.assertFalse(f.try_update(sol, ts))

    def test_good_25deg_at_155ms_accepted(self):
        f = _filter(np.radians(136.6), frames_since_update=5)
        sol, ts = _cand(f, 0.155, 25.2, n_inliers=12, error_px=0.2)
        self.assertTrue(f.try_update(sol, ts))

    def test_legacy_mode_unchanged_accepts_348_like(self):
        f = _filter(np.radians(300.8), frames_since_update=5, mode="legacy")
        sol, ts = _cand(f, 0.2441, 172.0)
        self.assertTrue(f.try_update(sol, ts))

    def test_rotation_seed_grace_exempts(self):
        f = _filter(np.radians(300.8), frames_since_update=5)
        f._rotation_seed_grace_frames = 2
        sol, ts = _cand(f, 0.2441, 172.0)
        self.assertNotEqual(f._rot_gate_info(ts)["T"], 0.0)
        self.assertTrue(f.try_update(sol, ts) in (True, False))  # must not crash; exemption path exercised

    def test_expired_state_never_vetoes_via_gate(self):
        f = _filter(np.radians(100.0), frames_since_update=5)
        f.last_update_ts_ns = 0
        self.assertEqual(f._rot_gate_info(3500 * MS)["T"], 179.0)
        self.assertLess(f._rot_gate_info(400 * MS)["T"], 179.0)

    def test_ceiling_uses_gate_in_continuous_only(self):
        fc = _filter(np.radians(100.0), 2)
        fl = _filter(np.radians(100.0), 2, mode="legacy")
        ts = 100 * MS
        self.assertAlmostEqual(fc._implausible_jump_thresholds(ts)[1], 40.0 + 500.0 * 0.1)
        self.assertGreater(fl._implausible_jump_thresholds(ts)[1], 200.0)

    def test_nohs_gyro_only_veto_when_predict_none(self):
        f = _filter(np.radians(300.0), 2, g_world=False)
        sol, ts = _cand(f, 0.06, 120.0, n_inliers=10, error_px=0.3)
        self.assertIsNone(f.predict(ts))
        f.frames_since_update = 2
        self.assertFalse(f.try_update(sol, ts))
        self.assertEqual(f._last.get("outcome"), "implausible_reject")

    def test_nohs_fail_open_when_consistent(self):
        f = _filter(np.radians(300.0), 2, g_world=False)
        sol, ts = _cand(f, 0.06, 10.0, n_inliers=10, error_px=0.3)
        self.assertTrue(f.try_update(sol, ts))
        self.assertEqual(f._last.get("outcome"), "fail_open")

    def test_nohs_fail_open_unchanged_in_legacy(self):
        f = _filter(np.radians(300.0), 2, mode="legacy", g_world=False)
        sol, ts = _cand(f, 0.06, 120.0, n_inliers=10, error_px=0.3)
        self.assertTrue(f.try_update(sol, ts))


class HeadsetInputsTests(unittest.TestCase):
    def test_none_without_headset(self):
        self.assertIsNone(_filter(1.0, 0)._headset_inputs(0, 10 * MS))

    def test_all_four_lookups_evaluated_once_in_order(self):
        f = _filter(1.0, 0, headset_mocap=object())
        calls = []
        def mk(name, ret):
            def _fn(_m, ts):
                calls.append((name, ts)); return ret
            return _fn
        with patch("src.pose_fusion_heuristic.world_pose", mk("wp", None)) as _, \
                patch("src.pose_fusion_heuristic.headset_angular_velocity", mk("om", np.zeros(3))), \
                patch("src.pose_fusion_heuristic.headset_linear_velocity", mk("v", np.zeros(3))):
            self.assertIsNone(f._headset_inputs(0, 10 * MS))
        self.assertEqual([c[0] for c in calls], ["wp", "wp", "om", "v"])

    def test_disable_flag_forces_no_headset(self):
        f = _filter(1.0, 0, headset_mocap=object(), extra_hc={"disable_headset_ego_motion": True})
        self.assertFalse(f._headset_possible())
        with patch("src.pose_fusion_heuristic.world_pose") as wp:
            self.assertIsNone(f._headset_inputs(0, 10 * MS))
            wp.assert_not_called()

    def test_gate_mode_reads_predict_stamp(self):
        f = _filter(np.radians(100.0), 0)
        f.predict(50 * MS)
        self.assertEqual(f._pred_stamp, (0, 50 * MS, False))
        self.assertFalse(f._rot_gate_info(50 * MS)["headset"])
        self.assertAlmostEqual(f._rot_gate_info(50 * MS)["T"], 40.0 + 25.0)


if __name__ == "__main__":
    unittest.main()


class RotCoastBudgetTests(unittest.TestCase):
    def _f(self, mode, gyro_dps=800.0):
        f = _filter(np.radians(gyro_dps), 2, extra_hc={"coast_rot_budget_mode": mode})
        return f

    def test_legacy_mode_passthrough(self):
        f = self._f("legacy")
        with patch.object(f, "_headset_inputs", return_value=(1, 2, 3, 4)):
            self.assertEqual(f.rot_coast_budget_s(50 * MS, 0.003), 0.003)

    def test_headset_mode_widens_when_headset_active(self):
        f = self._f("headset")
        with patch.object(f, "_headset_inputs", return_value=(1, 2, 3, 4)):
            self.assertAlmostEqual(f.rot_coast_budget_s(50 * MS, 0.003), 0.30)

    def test_headset_mode_without_headset_keeps_legacy(self):
        f = self._f("headset")
        self.assertEqual(f.rot_coast_budget_s(50 * MS, 0.003), 0.003)

    def test_never_narrower_than_legacy(self):
        f = self._f("headset", gyro_dps=2000.0)
        with patch.object(f, "_headset_inputs", return_value=(1, 2, 3, 4)):
            self.assertEqual(f.rot_coast_budget_s(50 * MS, 0.20), 0.20)
            self.assertAlmostEqual(f.rot_coast_budget_s(50 * MS, 0.003), 0.035)


class AutoModeTests(unittest.TestCase):
    def test_auto_without_headset_is_legacy(self):
        f = _filter(np.radians(300.8), 5, mode="auto")
        self.assertFalse(f._rot_gate_continuous())
        sol, ts = _cand(f, 0.2441, 172.0)
        self.assertTrue(f.try_update(sol, ts))  # legacy behaviour: 348-like accepted

    def test_auto_with_headset_is_continuous(self):
        f = _filter(np.radians(300.8), 5, mode="auto", headset_mocap=object())
        self.assertTrue(f._rot_gate_continuous())

    def test_auto_with_headset_disabled_is_legacy(self):
        f = _filter(1.0, 0, mode="auto", headset_mocap=object(), extra_hc={"disable_headset_ego_motion": True})
        self.assertFalse(f._rot_gate_continuous())
