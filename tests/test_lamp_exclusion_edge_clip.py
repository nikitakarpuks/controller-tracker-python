"""Regression coverage for src/blob_detector.py's _lamp_exclusion_contours
(fix landed 2026-09-11).

Bug: a removed lamp point sitting close to the real image's own edge had its
pad_px-radius exclusion circle clipped by the mask array's bounds, so
cv2.findContours' resulting contour boundary there was actually the ARRAY
edge, not the circle's true edge -- pointPolygonTest then measured a tiny
"depth" (how far inside the exclusion zone a pass-2 candidate sits) instead
of the real ~pad_px one. Confirmed on a real recording (cam0, right,
frame_range 1750-1800 relative frame 44, 640px-wide frames): a removed lamp
point at x=638.67 (only 1.33px from the 640px edge) got a measured
"depth" of just 0.33px against a pad_px=22 circle -- nowhere near
interior_edge_margin_px(10), so pass 2's own H1 re-detection at nearly the
identical position (638.74,152.47) was never excluded and resurfaced
un-filtered, identical to the point pass 1 had just correctly removed.

Fix: pad the mask canvas by pad_px on every side before drawing circles
(shifting draw coordinates by +pad_px), then shift the returned contours back
by -pad_px so callers see the same coordinate space as before -- a circle
near the real image edge is never clipped by the (now oversized) array.

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_lamp_exclusion_edge_clip
"""
import unittest

import cv2
import numpy as np

from src.blob_detector import _lamp_exclusion_contours


class ExclusionEdgeClipTests(unittest.TestCase):
    def test_point_near_image_edge_still_gets_full_pad_depth(self):
        """Real case: a single removed lamp point 1.33px from a 640px-wide
        frame's right edge, pad_px=22 (production's own
        pass2_exclusion_pad_px default)."""
        image_shape = (480, 640)
        point = (638.67, 152.33)
        pad_px = 22

        contours = _lamp_exclusion_contours([point], image_shape, pad_px)
        self.assertEqual(len(contours), 1)

        depth = cv2.pointPolygonTest(contours[0], point, True)
        self.assertGreater(depth, pad_px - 2.0,
                            f"a point at the center of its own {pad_px}px-radius exclusion "
                            f"circle must measure close to {pad_px}px deep (a few tenths off "
                            f"for cv2.circle's own pixel rasterization is expected), not "
                            f"clipped down to a couple px by the image boundary (got {depth:.2f})")

    def test_point_far_from_edge_matches_edge_case_depth(self):
        """Control: the SAME point, but far from any image boundary, must
        give (approximately) the same depth as the near-edge case above --
        proves the fix doesn't change behavior away from edges, only at them."""
        image_shape = (480, 2000)
        point = (1000.0, 152.33)
        pad_px = 22

        contours = _lamp_exclusion_contours([point], image_shape, pad_px)
        depth = cv2.pointPolygonTest(contours[0], point, True)
        self.assertGreater(depth, pad_px - 2.0)

    def test_returned_contour_coordinates_match_input_coordinate_space(self):
        """The padding is an internal implementation detail -- callers (which
        run pointPolygonTest against ORIGINAL image-space candidate
        centroids) must see contours already shifted back to that same
        space, not the padded canvas's own coordinates."""
        image_shape = (480, 640)
        point = (50.0, 50.0)
        pad_px = 22

        contours = _lamp_exclusion_contours([point], image_shape, pad_px)
        cnt = contours[0].reshape(-1, 2)
        # The contour must be centered near the ORIGINAL point (50,50), not
        # near (50+pad_px, 50+pad_px) -- i.e. still in the padded canvas's
        # own coordinate space.
        centroid = cnt.mean(axis=0)
        self.assertLess(np.linalg.norm(centroid - np.array([50.0, 50.0])), 1.0,
                         f"contour centroid {centroid} should be near the original "
                         f"point (50,50), not shifted by pad_px")

    def test_two_far_apart_points_yield_two_separate_contours(self):
        """Sanity: unrelated to the edge-clip fix, but confirms the padded
        canvas doesn't accidentally merge points that shouldn't be unioned."""
        image_shape = (480, 640)
        contours = _lamp_exclusion_contours([(50.0, 50.0), (500.0, 400.0)], image_shape, 22)
        self.assertEqual(len(contours), 2)


if __name__ == "__main__":
    unittest.main()
