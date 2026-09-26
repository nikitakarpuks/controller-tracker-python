"""blob_detection.streak_tail_trim: cut the fading tail off a motion-blur streak and keep its bright head."""
import unittest

import cv2
import numpy as np

from src.blob_detector import _detect_blobs, _trim_streak_tail

H, W = 100, 140


def _comet(head_xy=(50.0, 50.0), angle_deg=0.0, tail_len=25.0, peak=220.0, sigma=1.6):
    """Bright Gaussian head at head_xy followed by an exponentially fading tail along angle_deg (head = start of exposure)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = np.radians(angle_deg); ux, uy = np.cos(a), np.sin(a)
    t = (xx - head_xy[0]) * ux + (yy - head_xy[1]) * uy            # along the streak (>0 = tail side)
    d = -(xx - head_xy[0]) * uy + (yy - head_xy[1]) * ux           # across
    head = peak * np.exp(-(t ** 2 + d ** 2) / (2 * sigma ** 2))
    tail = np.where(t > 0, 0.45 * peak * np.exp(-t / (tail_len / 3.0)) * np.exp(-d ** 2 / (2 * 1.4 ** 2)), 0.0)
    return np.clip(head + tail, 0, 255).astype(np.uint8)


def _largest_contour(img, thr):
    m = (img >= thr).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


def _circ(c):
    a = cv2.contourArea(c.reshape(-1, 1, 2)); p = cv2.arcLength(c.reshape(-1, 1, 2), True)
    return 4 * np.pi * a / (p * p)


class TrimHelperTests(unittest.TestCase):
    def test_comet_is_cut_to_its_head(self):
        img = _comet(angle_deg=20.0)
        cnt = _largest_contour(img, 10)
        self.assertLess(_circ(cnt), 0.5, "fixture sanity: the whole streak must fail the circularity test")
        r = _trim_streak_tail(img, cnt, {})
        self.assertIsNotNone(r)
        hc, (cx, cy), area = r
        self.assertGreaterEqual(_circ(hc), 0.5)
        self.assertLess(area, 0.6 * cv2.contourArea(cnt.reshape(-1, 1, 2)))
        self.assertLess(np.hypot(cx - 50.0, cy - 50.0), 3.0, "head centroid stays near the true head")

    def test_head_is_independent_of_the_pixel_threshold(self):
        img = _comet(angle_deg=-35.0)
        a = _trim_streak_tail(img, _largest_contour(img, 8), {})
        b = _trim_streak_tail(img, _largest_contour(img, 14), {})
        self.assertIsNotNone(a); self.assertIsNotNone(b)
        self.assertLess(np.hypot(a[1][0] - b[1][0], a[1][1] - b[1][1]), 1.5)

    def test_round_blob_is_left_alone(self):
        img = np.zeros((H, W), np.uint8); cv2.circle(img, (50, 50), 3, 200, -1)
        self.assertIsNone(_trim_streak_tail(img, _largest_contour(img, 10), {}))

    def test_symmetric_oval_is_left_alone(self):
        img = np.zeros((H, W), np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        img = (200 * np.exp(-((xx - 60) ** 2 / (2 * 8.0 ** 2) + (yy - 50) ** 2 / (2 * 2.0 ** 2)))).astype(np.uint8)
        self.assertIsNone(_trim_streak_tail(img, _largest_contour(img, 10), {}), "peak in the middle -> no head/tail")

    def test_two_streaks_merged_end_to_end_are_left_alone(self):
        img = np.maximum(_comet((40.0, 50.0)), _comet((58.0, 50.0)))
        self.assertIsNone(_trim_streak_tail(img, _largest_contour(img, 10), {}))

    def test_short_or_out_of_range_elongation_is_left_alone(self):
        img = _comet(tail_len=4.0)
        self.assertIsNone(_trim_streak_tail(img, _largest_contour(img, 10), {"streak_tail_trim_min_length_px": 30.0}))


def _cfg(trim):
    return {"min_threshold": 10, "min_area": 1.0, "max_area": 2000.0, "min_circularity": 0.5,
            "min_streak_elongation": 2.0, "max_streak_elongation": 4.5, "min_streak_fill_ratio": 0.30,
            "min_split_dist": 4.0, "split_valley_ratio": 0.6, "outlier_factor": 3.0, "interior_edge_margin_px": 10.0,
            "streak_tail_trim": trim}


class DetectBlobsIntegrationTests(unittest.TestCase):
    def test_trim_moves_the_centroid_to_the_head_and_keeps_the_blob(self):
        img = _comet(angle_deg=0.0, tail_len=30.0)
        off = _detect_blobs(img, 10, 20, _cfg(False))
        on = _detect_blobs(img, 10, 20, _cfg(True))
        self.assertEqual(len(on[0]), 1, "the streak must survive as one blob with the trim on")
        d_on = np.hypot(on[0][0][0] - 50.0, on[0][0][1] - 50.0)
        self.assertLess(d_on, 3.0)
        if len(off[0]) == 1:   # legacy: kept via the elongation rescue -> centroid pulled toward the tail
            self.assertGreater(np.hypot(off[0][0][0] - 50.0, off[0][0][1] - 50.0), d_on + 2.0)

    def test_tail_no_longer_trips_the_area_ceiling_or_circularity_at_a_lower_threshold(self):
        # pass 2 style call: the ceiling is derived from the (trimmed) pass-1 head, the tail grows the component past it
        img = _comet(angle_deg=10.0, tail_len=30.0)
        head_area = _trim_streak_tail(img, _largest_contour(img, 15), {})[2]
        for thr in (6, 10):
            off = _detect_blobs(img, thr, 20, _cfg(False), max_area_override=2.0 * head_area)
            on = _detect_blobs(img, thr, 20, _cfg(True), max_area_override=2.0 * head_area)
            self.assertEqual(len(off[0]), 0, "fixture sanity: whole streak exceeds the ceiling")
            self.assertEqual(len(on[0]), 1, f"thr {thr}: head must be kept")
            self.assertLess(np.hypot(on[0][0][0] - 50.0, on[0][0][1] - 50.0), 3.0)

    def test_default_is_off(self):
        img = _comet()
        a = _detect_blobs(img, 10, 20, {k: v for k, v in _cfg(False).items() if k != "streak_tail_trim"})
        b = _detect_blobs(img, 10, 20, _cfg(False))
        np.testing.assert_array_equal(a[0], b[0])


if __name__ == "__main__":
    unittest.main()
