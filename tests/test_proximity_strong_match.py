"""Regression coverage for PoseSearcher._prox_is_strong -- proximity_search's
early-termination test, unified with brute-force's strong_found criteria
(error <= strong_match_error_px AND this camera's own pairs >=
strong_match_inliers).

Real case (static_easy recording, right_controller, cam3, ts=100830361704073,
mocap-confirmed): with the old error-only test (proximity_strong_match_px),
a hypothesis that dropped 2 of 5 ambiguous LEDs scored 0.37px on just 5
pairs and stopped the search after 13 hypotheses. The hypothesis it kept put
LED 19 onto LED 11's blob; that one off-axis pair rotated the solved pose
~15deg about the near-collinear axis of the other five LEDs (only 0.8-2.2px
of image motion for those five), and the resulting weak solve poisoned the
prediction for the next two frames, both lost.

The frame's actual proximity_search inputs (blobs, predicted pose) are
embedded below; camera/model/matching config are the project's real ones.
"""
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera
from src.controller import ControllerModel, create_leds_from_config
from src.load_config import load_json_config, load_yaml_config
from src.pose_search import PoseSearcher

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = REPO_ROOT / "data" / "cameras" / "calibration_basalt.json"
CTRL_PATH = REPO_ROOT / "data" / "controllers" / "right_controller_A85K6081930636R.json"
CONFIG_PATH = REPO_ROOT / "config" / "config.yml"

# Captured from the live run (right_controller, internal cam 3).
_BLOBS = np.array([[372.5, 210.5], [377.1, 216.8], [376.1, 228.8], [375.9, 242.4],
                   [389.2, 247.4], [384.0, 260.3], [401.7, 264.7], [392.8, 265.7]], dtype=np.float32)
_PRED_RVEC = np.array([0.912833154, -0.901643038, -2.074837446])
_PRED_TVEC = np.array([0.12433283, -0.017716747, 0.477071404])
_EXPANSION_PX = 12.5
# The five correct pairs (blob_idx, led_id): the wrong LED19->blob1 pair must NOT be part of the solve.
_GOOD_PAIRS = {(2, 9), (3, 15), (4, 4), (5, 17), (7, 6)}


def _rot_deg(rvec_a, rvec_b) -> float:
    Ra = cv2.Rodrigues(np.asarray(rvec_a, float).reshape(3, 1))[0]
    Rb = cv2.Rodrigues(np.asarray(rvec_b, float).reshape(3, 1))[0]
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1.0, 1.0))))


def _searcher(matching_overrides=None) -> PoseSearcher:
    cam = Camera(load_json_config(str(CALIB_PATH)), camera_idx=3)
    with open(CTRL_PATH) as f:
        model = ControllerModel(create_leds_from_config(json.load(f)), "right")
    matching = dict(load_yaml_config(str(CONFIG_PATH))["matching"])
    matching.update(matching_overrides or {})
    return PoseSearcher(cam, model, geometry_cfg={}, matching_cfg=matching)


class ProxIsStrongPredicateTests(unittest.TestCase):
    def setUp(self):
        self.ps = _searcher({"strong_match_error_px": 0.5, "strong_match_inliers": 6})

    def test_needs_both_error_and_pair_count(self):
        self.assertTrue(self.ps._prox_is_strong(0.3, 6))
        self.assertFalse(self.ps._prox_is_strong(0.3, 5), "too few pairs must never count as strong")
        self.assertFalse(self.ps._prox_is_strong(0.7, 8), "too much error must never count as strong")

    def test_boundaries_are_inclusive(self):
        self.assertTrue(self.ps._prox_is_strong(0.5, 6))
        self.assertFalse(self.ps._prox_is_strong(0.5001, 6))
        self.assertFalse(self.ps._prox_is_strong(0.5, 5))

    def test_shares_brute_force_thresholds(self):
        ps = _searcher({"strong_match_error_px": 1.0, "strong_match_inliers": 4})
        self.assertTrue(ps._prox_is_strong(0.9, 4))
        self.assertFalse(ps._prox_is_strong(1.1, 4))

    def test_proximity_strong_match_px_no_longer_controls_early_stop(self):
        ps = _searcher({"strong_match_error_px": 0.5, "strong_match_inliers": 6,
                        "proximity_strong_match_px": 5.0})
        self.assertFalse(ps._prox_is_strong(2.0, 8))


class Frame41RegressionTests(unittest.TestCase):
    """Real static_easy / right_controller / cam3 frame -- see module docstring."""

    def _run(self, overrides=None):
        ps = _searcher(overrides)
        return ps.proximity_search(_BLOBS, (_PRED_RVEC, _PRED_TVEC), expansion_px=_EXPANSION_PX)

    def test_fixed_search_does_not_stop_on_five_pair_fit(self):
        """Shipped config (strong_match_inliers=6): the 5-pair, 2-None 0.37px
        hypothesis is no longer 'strong', so the search reaches the better
        one. The pose then stays where the (mocap-verified good) prediction
        was; before the fix it was rotated ~15deg away."""
        sol = self._run()
        self.assertIsNotNone(sol)
        self.assertEqual({(int(b), int(l)) for b, l in sol["assignment"]}, _GOOD_PAIRS)
        self.assertLess(sol["error"], 0.3)
        self.assertLess(_rot_deg(_PRED_RVEC, sol["rvec"]), 3.0)
        self.assertLess(float(np.linalg.norm(np.asarray(sol["tvec"]).reshape(3) - _PRED_TVEC)) * 1000.0, 10.0)

    def test_control_old_behaviour_reproduces_the_bad_pose(self):
        """strong_match_inliers=5 lets the 5-pair hypothesis count as strong
        again -- exactly the old early stop (0.37px <= 0.5px). Guards against
        this regression test going vacuous: it must reproduce the ~15deg
        rotation error the fix removes."""
        sol = self._run({"strong_match_inliers": 5})
        self.assertIsNotNone(sol)
        self.assertGreater(_rot_deg(_PRED_RVEC, sol["rvec"]), 10.0)
        self.assertGreater(sol["error"], 0.5)


if __name__ == "__main__":
    unittest.main()
