"""rotation_seed_grace_vetted_deg: an exempt accept that AGREES with the gyro ends the rotation-seed grace at once;
one that disagrees keeps the old countdown (see RotationSeedGraceFramesTests for that 2026-09-13 case)."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tests.test_pose_fusion_heuristic import _make_filter, _solution, _stub_predict, _NS


class GraceVettedTests(unittest.TestCase):
    def _seeded(self, overrides=None):
        f = _make_filter(overrides)
        f.try_update(_solution(np.eye(3), np.zeros(3)), 0)
        f.predict = _stub_predict(np.eye(3), np.zeros(3))
        f.frames_since_update = 1000  # cold routing
        cold = {**_solution(np.eye(3), np.array([1.0, 0.0, 0.0])), "coverage_fallback": True}
        self.assertFalse(f.try_update(cold, 1 * _NS))
        self.assertTrue(f.try_update(cold, 2 * _NS))   # confirmed -> coverage-fallback seed
        self.assertEqual(f._rotation_seed_grace_frames, 2)
        return f

    def test_gyro_consistent_exempt_accept_clears_grace_and_gate_arms_next_frame(self):
        f = self._seeded()
        f.predict = _stub_predict(np.eye(3), np.array([1.0, 0.0, 0.0]))
        # N+1: rotation agrees with the gyro prediction (innovation ~0) -> vetted
        self.assertTrue(f.try_update(_solution(np.eye(3), np.array([1.0, 0.0, 0.0])), 3 * _NS))
        self.assertEqual(f._rotation_seed_grace_frames, 0)
        # N+2: a wildly different rotation must now be vetoed, state left unchanged
        p_before = f.p.copy()
        f.predict = _stub_predict(np.eye(3), np.array([1.0, 0.0, 0.0]))
        ok = f.try_update(_solution(Rotation.from_euler('z', 133, degrees=True).as_matrix(),
                                     np.array([1.0, 0.0, 0.0])), 4 * _NS)
        self.assertFalse(ok)
        np.testing.assert_allclose(f.p, p_before)

    def test_disagreeing_exempt_accept_keeps_countdown(self):
        f = self._seeded()
        f.predict = _stub_predict(np.eye(3), np.array([1.0, 0.0, 0.0]))
        self.assertTrue(f.try_update(_solution(Rotation.from_euler('z', 150, degrees=True).as_matrix(),
                                                np.array([1.0, 0.0, 0.0])), 3 * _NS))
        self.assertEqual(f._rotation_seed_grace_frames, 1)

    def test_borderline_innovation_uses_threshold(self):
        f = self._seeded()
        f.predict = _stub_predict(np.eye(3), np.array([1.0, 0.0, 0.0]))
        f.try_update(_solution(Rotation.from_euler('z', 25, degrees=True).as_matrix(), np.array([1.0, 0.0, 0.0])), 3 * _NS)
        self.assertEqual(f._rotation_seed_grace_frames, 1, "25 deg > 20 deg vetted threshold -> old countdown")

    def test_disabled_keeps_old_behaviour(self):
        f = self._seeded({"rotation_seed_grace_vetted_deg": 0.0})
        f.predict = _stub_predict(np.eye(3), np.array([1.0, 0.0, 0.0]))
        f.try_update(_solution(np.eye(3), np.array([1.0, 0.0, 0.0])), 3 * _NS)
        self.assertEqual(f._rotation_seed_grace_frames, 1)


if __name__ == "__main__":
    unittest.main()
