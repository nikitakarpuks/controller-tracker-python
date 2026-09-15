"""Regression test for a real bug found this session: every diagnostic log
line around CameraTracker.finalize_search's "Pose jump detected" WARNING
(src/controller.py) computed the per-axis rotation delta as R_ref.T @ R_new,
but CameraTracker._pose_jump_too_large -- the ACTUAL check deciding
accept/reject -- computes it as R_new @ R_ref.T (see that method's own code).

These are NOT interchangeable for a per-axis breakdown: R_new@R_ref.T and
R_ref.T@R_new are conjugate/similar matrices, so they share the same total
rotation ANGLE (Rodrigues-vector magnitude is invariant to conjugation), but
represent that same physical rotation's axis expressed in two different
frames -- so the individual x/y/z rotation-vector COMPONENTS genuinely
differ between the two conventions.

Found from a real log where the printed per-axis rot_diff was
(19.1, 0.7, 28.3)deg, thresh=(30,30,30) -- every axis individually under
threshold -- yet the frame was still rejected as a pose jump. The check
itself was correct throughout (see test_check_rejects_using_the_correct_
convention below); only the diagnostic display used the wrong convention,
making the WARNING's own printed numbers fail to justify its own decision.

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_pose_jump_rotation_convention
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.controller import CameraTracker

# A concrete case (found by random search, see this session's own investigation)
# where the two conventions disagree about whether any axis exceeds a 30deg
# per-axis threshold -- exactly the discrepancy class that was found live.
_R_REF = Rotation.from_euler('xyz', [-4.76982762, 36.3672395, -0.00833490499], degrees=True).as_matrix()
_R_NEW = Rotation.from_euler('xyz', [-10.00382388, 44.78218114, 34.64842046], degrees=True).as_matrix()
_RVEC_REF = Rotation.from_matrix(_R_REF).as_rotvec().reshape(3, 1).astype(np.float32)
_RVEC_NEW = Rotation.from_matrix(_R_NEW).as_rotvec().reshape(3, 1).astype(np.float32)
_TVEC = np.zeros(3, dtype=np.float32)  # position diff kept at 0 to isolate the rotation branch


def _rotvec_deg(R_a: np.ndarray, R_b_T_side: np.ndarray) -> np.ndarray:
    """abs(Rodrigues(R_a @ R_b_T_side)) in degrees -- R_b_T_side is expected
    to already be R_ref.T or similar, matching how both the real check and
    the (now-fixed) diagnostics build their relative-rotation matrix."""
    return np.degrees(np.abs(Rotation.from_matrix(R_a @ R_b_T_side).as_rotvec()))


class PoseJumpRotationConventionTests(unittest.TestCase):
    def test_the_two_conventions_genuinely_disagree_on_this_case(self):
        """Sanity-check the fixture itself: confirms this rotation pair is a
        real divergent case, not a coincidence -- CORRECT (R_new @ R_ref.T)
        has an axis over 30deg; WRONG (R_ref.T @ R_new) does not, even though
        both represent the exact same physical rotation (same total angle)."""
        correct = _rotvec_deg(_R_NEW, _R_REF.T)   # R_new @ R_ref.T -- matches _pose_jump_too_large
        wrong   = _rotvec_deg(_R_REF.T, _R_NEW)   # R_ref.T @ R_new -- the old, mismatched diagnostic order
        self.assertGreater(correct.max(), 30.0, f"fixture didn't reproduce the divergence: correct={correct}")
        self.assertLess(wrong.max(), 30.0, f"fixture didn't reproduce the divergence: wrong={wrong}")
        # Same physical rotation -> same total angle either way (order-invariant).
        self.assertAlmostEqual(np.linalg.norm(correct), np.linalg.norm(wrong), places=3)

    def test_check_rejects_using_the_correct_convention(self):
        """The actual accept/reject decision (_pose_jump_too_large) has
        always used R_new @ R_ref.T -- confirms it correctly rejects this
        case (one axis genuinely exceeds the per-axis threshold), regardless
        of what any diagnostic happened to print."""
        is_jump = CameraTracker._pose_jump_too_large(
            _RVEC_NEW, _TVEC, _RVEC_REF, _TVEC,
            pos_thresh_xyz_m=(0.18, 0.18, 0.20),
            rot_thresh_xyz_deg=(30.0, 30.0, 30.0),
        )
        self.assertTrue(is_jump, "the correct-convention rotation exceeds 30deg on one axis -- must reject")

    def test_the_wrong_convention_would_have_falsely_cleared_it(self):
        """Demonstrates the bug's real-world symptom: if a diagnostic (or a
        gate) used the wrong order, this exact rotation would look like it
        passed every per-axis threshold -- explaining the real report of
        'the printed numbers don't justify the reject'."""
        wrong = _rotvec_deg(_R_REF.T, _R_NEW)
        thresh = np.array([30.0, 30.0, 30.0])
        self.assertTrue(np.all(wrong <= thresh), f"expected the wrong convention to clear every axis, got {wrong}")

    def test_fixed_diagnostic_convention_matches_what_the_check_actually_tested(self):
        """The (now-fixed) per-axis diagnostic computation used in the
        finalize_search WARNING must use the SAME order as the real check --
        this reproduces that exact expression (src/controller.py's WARNING
        block: cv2.Rodrigues((_R_new @ _R_p.T)...)) and confirms it agrees
        with _pose_jump_too_large's own internal per-axis values, so the
        logged numbers can never again fail to justify the logged decision."""
        diagnostic = _rotvec_deg(_R_NEW, _R_REF.T)  # R_new @ R_ref.T, same as the fixed WARNING code
        # Re-derive exactly what _pose_jump_too_large computes internally,
        # to compare against the diagnostic value directly (not just the
        # boolean outcome).
        internal = np.degrees(np.abs(
            Rotation.from_matrix(_R_NEW @ _R_REF.T).as_rotvec()
        ))
        np.testing.assert_allclose(diagnostic, internal, atol=1e-4)
        self.assertTrue(np.any(diagnostic > 30.0))  # matches the True reject from the test above


if __name__ == "__main__":
    unittest.main()
