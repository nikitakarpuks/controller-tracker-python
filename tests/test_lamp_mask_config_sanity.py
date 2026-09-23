"""Config-sanity regression test for blob_detection.lamp_blob_filter.static_lamp_mask
in config/config.yml -- catches the shipped-value class of bug found in an
independent 2026-09-14 code review: `min_hits_to_exclude: 1` and
`max_brightness: 255` both silently defeated their own documented safety
purpose (see src/lamp_region_memory.py's own __init__ comments and
config.yml's own comments on these two keys for the full story) without any
test noticing, because the two normal test suites for this feature
(tests/test_lamp_region_memory.py, tests/test_lamp_pass2_merged_blob_context.py)
construct their own cfg dicts directly rather than reading the real shipped
config.yml. This file reads THE REAL config.yml -- it exists specifically to
catch a production config value drifting back into a no-op again, which a
class-constructor-level assertion cannot do without also breaking legitimate
unit tests that deliberately use relaxed values to isolate unrelated behavior.

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_lamp_mask_config_sanity
"""
import unittest
from pathlib import Path

from src.load_config import load_yaml_config

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yml"


class StaticLampMaskConfigSanityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_yaml_config(str(_CONFIG_PATH))
        cls.mask_cfg = cls.cfg["blob_detection"]["lamp_blob_filter"]["static_lamp_mask"]
        cls.lamp_cfg = cls.cfg["blob_detection"]["lamp_blob_filter"]

    def test_min_hits_to_exclude_actually_requires_reinforcement(self):
        # LampRegion.hits starts at 1 (src/lamp_region_memory.py's own
        # dataclass default) -- a value of 1 (or less) here makes
        # `hits < min_hits_to_exclude` false on a region's very first frame,
        # i.e. no reinforcement is ever actually required. Shipped as 1 for
        # a time despite this exact class's own test suite
        # (tests/test_lamp_region_memory.py's ExclusionGateTests-equivalent)
        # validating 3 as the intended, protective value.
        val = self.mask_cfg.get("min_hits_to_exclude", 3)
        self.assertGreaterEqual(
            val, 2,
            "static_lamp_mask.min_hits_to_exclude must be >= 2 -- LampRegion.hits "
            "starts at 1, so 1 makes the 'wait for reinforcement before excluding "
            "anything' safety net a complete no-op.")

    def test_max_brightness_is_a_real_discriminating_threshold_if_set(self):
        # Blob brightness (max_pix) is an 8-bit pixel value, 0-255. A
        # static_lamp_mask.max_brightness of 255 makes `max_pix <= 255`
        # unconditionally true, so the "spare a real, bright controller LED
        # that wanders through a remembered lamp region" guard can never
        # fire -- shipped this way for a time. The key is now normally
        # absent entirely (falls back to lamp_blob_filter.max_brightness,
        # checked below), but guard against it being re-added with a
        # meaningless value.
        if "max_brightness" in self.mask_cfg:
            self.assertLess(
                self.mask_cfg["max_brightness"], 255,
                "static_lamp_mask.max_brightness must be < 255 (an 8-bit pixel "
                "max) -- 255 makes the real-LED brightness safety guard a no-op.")

    def test_max_brightness_fallback_source_is_itself_a_real_threshold(self):
        # Whether or not static_lamp_mask sets its own max_brightness,
        # src/blob_detector.py falls back to lamp_blob_filter.max_brightness
        # -- confirm THAT value is itself meaningful, since a no-op fallback
        # would silently reintroduce the same bug one level up.
        self.assertLess(
            self.lamp_cfg.get("max_brightness", 210.0), 255,
            "blob_detection.lamp_blob_filter.max_brightness must be < 255 -- "
            "static_lamp_mask falls back to this value when it sets no "
            "max_brightness of its own.")

    def test_expire_after_no_support_frames_is_positive(self):
        val = self.mask_cfg.get("expire_after_no_support_frames", 10)
        self.assertGreater(val, 0, "a non-positive value would expire a region "
                            "the instant it fails to sustain even once, defeating "
                            "the documented 'short grace period' intent.")

    def test_sustain_min_points_is_positive(self):
        val = self.mask_cfg.get("sustain_min_points", 3)
        self.assertGreater(val, 0, "0 (or less) would sustain every region "
                            "unconditionally regardless of any real support.")


if __name__ == "__main__":
    unittest.main()
