"""Unit coverage for the LED visibility scoring proximity_search actually
gates on (src/_visibility.py's _visible_mask, called from
PoseSearcher.proximity_search at src/pose_search.py:775-796) and for whether
proximity_vis_score_threshold (config/config.yml, matching:) has any real
effect once combined with led_facing_angle_deg.

Calls _visible_mask directly with synthetic, controlled LED
positions/normals/angles rather than a full PoseSearcher (which needs a real
camera + a full ControllerGeometry fit) -- this is the exact same function
proximity_search calls, with the exact same parameter (facing_threshold_deg)
and the exact same comparison proximity_search itself makes
(vis_scores >= proximity_vis_score_threshold, see pose_search.py:796) to
decide which LEDs are "visible" for matching. cam_K=None skips the in-frame
check (2b); a minimal geom stub with boxes=[]/cylinders=[]/is_inner=None
skips the frustum (3) and handle-body (4) occlusion checks -- isolating
purely to check 2 (the facing-angle score), which is what
led_facing_angle_deg/proximity_vis_score_threshold together control.

Motivated by a real investigation this session: led_facing_angle_deg was
found misconfigured at 90.0 (see config.yml's own comment), which made the
facing score degenerate (every LED in the front hemisphere scored exactly
1.0, so proximity_vis_score_threshold had nothing left to filter). These
tests are the regression check for that finding, and directly answer
"is proximity_vis_score_threshold working" for any future config value.

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_visibility_scoring
"""
import math
import unittest
from types import SimpleNamespace

import numpy as np

from src._visibility import _visible_mask

# Same live values as config/config.yml's matching: block.
LIVE_PROXIMITY_VIS_SCORE_THRESHOLD = 0.95
LIVE_LED_FACING_ANGLE_DEG = 75.0
OLD_BUGGY_LED_FACING_ANGLE_DEG = 90.0

_EMPTY_GEOM = SimpleNamespace(boxes=[], cylinders=[], is_inner=None)


def _led_at_angle(angle_deg: float, depth_m: float = 1.0):
    """One synthetic LED, R=I, tvec=(0,0,depth_m) (camera looking straight at
    it along +z), with its normal tilted so _visible_mask's own facing-angle
    computation (src/_visibility.py: angle_deg = degrees(arccos(clip(-dot,
    -1,1)))) comes out to exactly `angle_deg`. Returns (R, tvec, positions,
    normals) ready to pass straight into _visible_mask."""
    R = np.eye(3, dtype=np.float32)
    tvec = np.array([0.0, 0.0, depth_m], dtype=np.float32)
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    a = math.radians(angle_deg)
    normals = np.array([[math.sin(a), 0.0, -math.cos(a)]], dtype=np.float32)
    return R, tvec, positions, normals


def _score_at_angle(angle_deg: float, facing_threshold_deg: float) -> float:
    R, tvec, positions, normals = _led_at_angle(angle_deg)
    scores = _visible_mask(R, tvec, positions, normals, _EMPTY_GEOM,
                            facing_threshold_deg=facing_threshold_deg)
    return float(scores[0])


class FacingScoreFormulaTests(unittest.TestCase):
    """The raw scoring function _visible_mask computes -- independent of
    proximity_vis_score_threshold, this is what feeds it."""

    def test_directly_facing_led_scores_1(self):
        self.assertAlmostEqual(_score_at_angle(0.0, LIVE_LED_FACING_ANGLE_DEG), 1.0, places=5)

    def test_led_within_threshold_angle_scores_1(self):
        # 60deg < 75deg threshold -> still full score, per the "score=1 while
        # angle < threshold" branch of the formula.
        self.assertAlmostEqual(_score_at_angle(60.0, LIVE_LED_FACING_ANGLE_DEG), 1.0, places=5)

    def test_led_past_threshold_angle_ramps_down(self):
        score = _score_at_angle(85.0, LIVE_LED_FACING_ANGLE_DEG)
        self.assertLess(score, 1.0)
        self.assertAlmostEqual(score, LIVE_LED_FACING_ANGLE_DEG / 85.0, places=5)

    def test_led_at_90deg_scores_0(self):
        self.assertEqual(_score_at_angle(90.0, LIVE_LED_FACING_ANGLE_DEG), 0.0)

    def test_led_behind_controller_scores_0(self):
        # angle > 90 (normal pointing away from the camera entirely) --
        # facing_score's own gate is `angle_deg < 90.0`.
        self.assertEqual(_score_at_angle(135.0, LIVE_LED_FACING_ANGLE_DEG), 0.0)


class ThresholdInteractionTests(unittest.TestCase):
    """The actual thing this session investigated: does
    proximity_vis_score_threshold meaningfully exclude anything, and does
    that depend on led_facing_angle_deg being set to a real (non-degenerate)
    value? Reproduces proximity_search's own comparison
    (pose_search.py:796): vis_mask = vis_scores >= proximity_vis_score_threshold.
    """

    def test_old_buggy_facing_angle_defeats_the_threshold(self):
        """Regression test for the actual bug found this session: at
        led_facing_angle_deg=90 (the old default), EVERY LED short of
        literally edge-on gets a perfect 1.0 score -- proximity_vis_score_threshold
        has nothing left to filter, no matter how grazing the real angle is."""
        for angle in (0.0, 45.0, 80.0, 89.0, 89.9):
            score = _score_at_angle(angle, OLD_BUGGY_LED_FACING_ANGLE_DEG)
            self.assertEqual(
                score, 1.0,
                f"angle={angle}: expected the degenerate old threshold to still score 1.0, got {score}"
            )
            self.assertGreaterEqual(
                score, LIVE_PROXIMITY_VIS_SCORE_THRESHOLD,
                f"angle={angle}: proximity_vis_score_threshold={LIVE_PROXIMITY_VIS_SCORE_THRESHOLD} "
                f"should have been powerless to exclude this grazing-angle LED under the old bug"
            )

    def test_fixed_facing_angle_lets_the_threshold_actually_exclude_grazing_leds(self):
        """With the fix (led_facing_angle_deg=75), a genuinely grazing LED
        (80deg+) now scores below proximity_vis_score_threshold and gets
        correctly excluded -- the threshold is doing real work again."""
        excluded_angles = []
        included_angles = []
        for angle in (0.0, 30.0, 60.0, 75.0, 78.0, 80.0, 85.0, 89.0):
            score = _score_at_angle(angle, LIVE_LED_FACING_ANGLE_DEG)
            visible = score >= LIVE_PROXIMITY_VIS_SCORE_THRESHOLD
            (included_angles if visible else excluded_angles).append(angle)
        # Comfortably-facing LEDs still included.
        self.assertIn(0.0, included_angles)
        self.assertIn(60.0, included_angles)
        # Grazing LEDs now genuinely excluded -- this is the exact behavior
        # that was silently broken before the config fix.
        self.assertIn(85.0, excluded_angles)
        self.assertIn(89.0, excluded_angles)

    def test_threshold_comparison_is_inclusive_at_the_boundary(self):
        """proximity_search's own comparison is >=, matching its config
        comment ('include LEDs with vis score >= this') -- verify an LED
        landing exactly on the threshold is included, not excluded."""
        # angle_deg such that led_facing_angle_deg/angle_deg == threshold exactly.
        angle_deg = LIVE_LED_FACING_ANGLE_DEG / LIVE_PROXIMITY_VIS_SCORE_THRESHOLD
        score = _score_at_angle(angle_deg, LIVE_LED_FACING_ANGLE_DEG)
        self.assertAlmostEqual(score, LIVE_PROXIMITY_VIS_SCORE_THRESHOLD, places=5)
        self.assertTrue(score >= LIVE_PROXIMITY_VIS_SCORE_THRESHOLD)  # the exact live comparison

    def test_raising_the_threshold_excludes_more_leds_at_the_same_angle(self):
        """Holding led_facing_angle_deg fixed, a stricter
        proximity_vis_score_threshold should exclude LEDs a looser one would
        have kept -- confirms the threshold isn't a no-op knob."""
        angle = 80.0
        score = _score_at_angle(angle, LIVE_LED_FACING_ANGLE_DEG)
        self.assertTrue(score >= 0.90)   # kept under a looser bar
        self.assertFalse(score >= 0.99)  # excluded under a stricter one


if __name__ == "__main__":
    unittest.main()
