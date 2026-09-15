"""Tests for src/lamp_blob_filter.py -- the single-frame lamp-fixture blob
filter's deterministic global line finder plus its brightness guard.

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_lamp_blob_filter
"""
import time
import unittest

import numpy as np

from src.blob_detector import BlobResult
from src.lamp_blob_filter import detect_lamp_blobs, restrict_dim_context_to_kept_neighborhood

_DEFAULT_CFG = {
    "max_brightness": 60.0,
    "max_line_residual_px": 2.0,
    "spacing_min_px": 4.0,
    "spacing_max_px": 25.0,
    "min_points": 6,
}


def _square_contour(center, half=0.5):
    cx, cy = center
    return np.array([[cx - half, cy - half], [cx + half, cy - half],
                      [cx + half, cy + half], [cx - half, cy + half]], dtype=np.float32)


def _circle_contour(center, radius, n=12):
    cx, cy = center
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)], axis=1).astype(np.float32)


def _make_blobs(points, brightnesses, areas=None):
    """points: list of (x,y); brightnesses: list of float, same length.
    Every blob gets a contour sized to `areas` (default ~1px^2), well within
    the filter's default area_min/area_max range."""
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    if areas is None:
        areas = [1.0] * n
    radii = [max(np.sqrt(a / np.pi), 0.3) for a in areas]
    return BlobResult(
        centroids=points.copy(),
        radii=np.asarray(radii, dtype=np.float32),
        brightnesses=np.asarray(brightnesses, dtype=np.float32),
        contours=[_circle_contour(p, r) for p, r in zip(points, radii)],
    )


def _lamp_row(n=10, spacing=7.0, y=100.0, x0=100.0, brightness=20.0):
    points = [(x0 + i * spacing, y) for i in range(n)]
    return points, [brightness] * n


def _lamp_row_with_gap(n=10, spacing=7.0, gap_after=5, gap_px=20.6, y=100.0, x0=100.0, brightness=20.0):
    """A real lamp row confirmed on this project's own recordings: mostly
    ~7px spacing, with one larger physical gap partway along (measured up
    to ~20.6px Euclidean on cam3's own frame 0) -- the exact case the old
    chain-grower could not bridge without also mis-chaining across the
    fixture's second row."""
    points = []
    x = x0
    for i in range(n):
        points.append((x, y))
        x += gap_px if i == gap_after else spacing
    return points, [brightness] * n


class CleanLampRowTests(unittest.TestCase):
    def test_clean_lamp_row_is_filtered(self):
        points, brightness = _lamp_row()
        blobs = _make_blobs(points, brightness)
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        self.assertFalse(result.keep_mask.any(), "every blob in a clean lamp row should be removed")
        self.assertEqual(len(result.rejected_candidates), 1)
        self.assertEqual(len(result.guarded_candidates), 0)

    def test_lamp_row_with_real_physical_gap_is_filtered(self):
        """The core motivating case for this rewrite: a single ~19px gap
        partway along the row (real, confirmed on this project's own
        recordings) must not stop the whole row from being recognized and
        removed as one candidate."""
        points, brightness = _lamp_row_with_gap()
        blobs = _make_blobs(points, brightness)
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        self.assertFalse(result.keep_mask.any(), "the row must be removed even across a real physical gap")
        self.assertEqual(len(result.rejected_candidates), 1)
        self.assertEqual(len(result.rejected_candidates[0].blob_indices), len(points))

    def test_two_parallel_rows_both_removed(self):
        """The fixture's real two-row structure -- each row independently
        satisfies min_points, no pairing between them is required."""
        row_a, bright_a = _lamp_row(y=100.0)
        row_b, bright_b = _lamp_row(y=106.0)  # ~6px perpendicular offset, matches real fixture
        points = row_a + row_b
        brightness = bright_a + bright_b
        blobs = _make_blobs(points, brightness)
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        self.assertFalse(result.keep_mask.any(), "both rows should be removed")
        self.assertEqual(len(result.rejected_candidates), 2)


class RandomScatterTests(unittest.TestCase):
    def test_random_scatter_is_untouched(self):
        rng = np.random.RandomState(0)
        points = rng.uniform(0, 500, size=(30, 2))
        blobs = _make_blobs(points, [20.0] * 30)
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        self.assertTrue(result.keep_mask.all(), "unstructured scatter should never be flagged")
        self.assertEqual(len(result.rejected_candidates), 0)

    def test_many_random_scatters_never_flagged(self):
        """A single fixed seed proves little on its own -- sweep several to
        reduce the chance this test only happens to pass by luck."""
        for seed in range(20):
            rng = np.random.RandomState(seed)
            points = rng.uniform(0, 500, size=(30, 2))
            blobs = _make_blobs(points, [20.0] * 30)
            result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
            self.assertTrue(result.keep_mask.all(), f"seed={seed}: unstructured scatter must never be flagged")


class PointCountAndSpacingGuardTests(unittest.TestCase):
    """These encode the actual safety argument for this design (see
    src/lamp_blob_filter.py's module docstring): a real controller-LED ring,
    at any pose, either shows too FEW near-collinear points, or bunches them
    too TIGHTLY (< spacing_min_px) -- never a long run of realistically
    (4-25px) spaced points. min_points and the spacing bounds are the actual
    guard; max_line_residual_px is generous by comparison."""

    def test_short_straight_run_below_min_points_is_never_removed(self):
        points, brightness = _lamp_row(n=5)  # perfectly straight, correctly spaced, but < min_points=6
        blobs = _make_blobs(points, brightness)
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        self.assertTrue(result.keep_mask.all(), "fewer than min_points must never be removed, no matter how straight")

    def test_seven_point_row_with_one_outlier_is_still_removed(self):
        """Real regression case (frame_range 200-300, relative frame 32 of
        this project's own recording): a genuine 7-point lamp row plus one
        non-collinear outlier point -- 8 combined points total, but only 7
        of them actually lie on the row. min_points=8 rejected this outright
        (7 < 8); min_points=6 must recognize and remove the 7 real ones."""
        points, brightness = _lamp_row(n=7)
        outlier = [(points[-1][0] + 8.0, points[-1][1] - 4.0)]  # breaks the line's own direction
        blobs = _make_blobs(points + outlier, brightness + [20.0])
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        self.assertFalse(result.keep_mask[:7].any(), "the real 7-point row must be removed")

    def test_tightly_packed_cluster_is_never_removed(self):
        """Mimics a real controller ring viewed near edge-on: many points,
        but bunched far tighter (~2px) than spacing_min_px -- this is what a
        real ring's silhouette-edge LEDs actually look like in projection,
        per this module's own controller-pose simulation."""
        n = 12
        points = [(100.0 + i * 2.0, 200.0) for i in range(n)]
        blobs = _make_blobs(points, [20.0] * n)
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        self.assertTrue(result.keep_mask.all(), "a tightly-packed cluster must never be removed")

    def test_gap_too_large_breaks_the_run(self):
        """A gap well past spacing_max_px (not a real fixture's ~19-20px
        physical gap, but something unrelated far off to one side) must not
        pull in a second, unconnected cluster of points."""
        row, brightness = _lamp_row(n=8)
        far_away = [(500.0, 500.0), (510.0, 500.0)]
        points = row + far_away
        blobs = _make_blobs(points, brightness + [20.0, 20.0])
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        n_row = len(row)
        self.assertFalse(result.keep_mask[:n_row].any(), "the 8-point row alone still qualifies and should be removed")
        self.assertTrue(result.keep_mask[n_row:].all(), "the unrelated far-away pair must be untouched")


class AreaBoundTests(unittest.TestCase):
    """Regression for a bug found 2026-09-10: area_max (unlike spacing/
    min_points/residual) never had a documented safety derivation -- it was
    just excluding real lamp elements outright, before they ever reached the
    line-finder, whenever their blob area happened to exceed it."""

    def test_real_column_with_larger_trailing_blobs_is_fully_removed(self):
        """Real case (cam3, right, frame_range 600-700 relative frame 1): a
        genuine single-column 14-point lamp fixture where 8 points have area
        7-10px^2 and the other 6 have area 13.5-19.5px^2 -- all 14 are
        collinear (real max residual ~1.1px) and evenly spaced (~7-9px), but
        the old area_max=15.0 silently dropped the 6 larger ones from area_ok
        entirely, so they could never be removed no matter how obviously they
        lined up with the other 8."""
        points, brightness = _lamp_row(n=14, spacing=8.0, x0=370.0, y=430.0, brightness=13.0)
        areas = [8.0, 8.0, 9.0, 10.0, 10.0, 7.0, 8.5, 9.0,
                 17.0, 19.5, 15.5, 19.0, 14.5, 13.5]
        blobs = _make_blobs(points, brightness, areas=areas)
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        self.assertFalse(result.keep_mask[:8].any(), "the 8 smaller-area (in-range) points "
                                                       "still get found and removed as their own line")
        self.assertTrue(result.keep_mask[8:].all(),
                         "with the OLD area_max=15.0 default, the 6 larger-area points are "
                         "invisible to the line-finder entirely and survive unfiltered -- "
                         "this asserts today's buggy baseline; the fix lives in config.yml's "
                         "area_max=25.0 override, verified separately since detect_lamp_blobs' "
                         "own code default is intentionally left at 15.0 for this baseline")

    def test_same_column_fully_removed_once_area_max_widened(self):
        points, brightness = _lamp_row(n=14, spacing=8.0, x0=370.0, y=430.0, brightness=13.0)
        areas = [8.0, 8.0, 9.0, 10.0, 10.0, 7.0, 8.5, 9.0,
                 17.0, 19.5, 15.5, 19.0, 14.5, 13.5]
        blobs = _make_blobs(points, brightness, areas=areas)
        result = detect_lamp_blobs(blobs, {**_DEFAULT_CFG, "area_min": 0.4, "area_max": 25.0})
        self.assertFalse(result.keep_mask.any(),
                          "widening area_max to cover the real observed blob sizes must "
                          "remove the entire 14-point column, not just the smaller-area half")

    def test_second_real_case_with_even_larger_blobs_needs_area_max_50(self):
        """Real case (cam1, right, frame_range 1785-1790, four separate frames,
        bbox ~[510,90,635,160]): the SAME underestimate recurred with a second
        real fixture whose own 6 real elements (not 8+6 like the first case --
        this fixture is only 6 points total) have areas of 28.5-36.0px^2 --
        all individually-legitimate, correctly-circular kept blobs, all
        directly in-line (real spacing ~9-11px). Only 2 of those 6 (20.5 and
        23.5) happened to already be under the OLD area_max=25.0, leaving just
        2 area_ok points -- below min_points=6, so the OLD threshold recognized
        NOTHING here at all (unlike the first case, where 8 OTHER, separate
        small-area points on the same row still formed their own qualifying
        line while the 6 large ones survived unfiltered alongside it)."""
        points, brightness = _lamp_row(n=6, spacing=10.0, x0=500.0, y=430.0, brightness=15.0)
        areas = [28.5, 36.0, 30.0, 20.5, 23.5, 29.0]
        blobs = _make_blobs(points, brightness, areas=areas)

        result_old = detect_lamp_blobs(blobs, {**_DEFAULT_CFG, "area_min": 0.4, "area_max": 25.0})
        self.assertTrue(result_old.keep_mask.all(),
                         "with the OLD area_max=25.0, only 2 of these 6 real points "
                         "(20.5, 23.5) are area_ok -- below min_points=6, so nothing "
                         "gets recognized/removed at all")

        result_new = detect_lamp_blobs(blobs, {**_DEFAULT_CFG, "area_min": 0.4, "area_max": 50.0})
        self.assertFalse(result_new.keep_mask.any(),
                          "widening area_max to 50.0 must remove the entire 6-point column, "
                          "including the four 28.5-36.0px^2 points that used to be invisible")


class ControllerRingSafetyTests(unittest.TestCase):
    """Direct regression test against this project's own real controller LED
    geometry and camera model -- the actual empirical basis for this
    filter's safety margin (see src/lamp_blob_filter.py's module docstring).
    Systematically sweeps edge-on/tangential controller poses (the ring
    plane forced to contain the camera's view axis -- the flattest, most
    lamp-row-like projection a real ring can ever produce) and asserts the
    filter never removes anything from the projected, visible LEDs."""

    def test_edge_on_controller_poses_never_removed(self):
        try:
            import json
            from scipy.spatial.transform import Rotation as Rot
            from src.camera import Camera
            from src.controller import create_leds_from_config
            from src.load_config import load_yaml_config, load_json_config
        except Exception as e:  # pragma: no cover - environment-dependent
            self.skipTest(f"camera/controller stack unavailable: {e}")

        config = load_yaml_config("config/config.yml")
        calib_cfg = load_json_config(config["cameras"]["intrinsics_path"])
        cam = Camera(calib_cfg, camera_idx=3,
                     extrinsics_convention=config["cameras"].get("extrinsics_convention", "T_imu_cam"))

        with open(config["controllers"]["right_controller"]["config_path"]) as f:
            d = json.load(f)
        leds = create_leds_from_config(d)
        positions = np.array([l.position for l in leds], dtype=np.float64)
        normals = np.array([l.normal for l in leds], dtype=np.float64)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        facing_cos = np.cos(np.radians(float(config["matching"].get("led_facing_angle_deg", 70.0))))

        centroid3d = positions.mean(axis=0)
        _, _, vt3 = np.linalg.svd(positions - centroid3d)
        ring_normal = vt3[2]
        ring_normal /= np.linalg.norm(ring_normal)

        view_axis = np.array([0.0, 0.0, 1.0])
        perp_dir = np.array([1.0, 0.0, 0.0])

        def rotation_aligning(a, b):
            a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
            v = np.cross(a, b); c = np.dot(a, b)
            if np.linalg.norm(v) < 1e-8:
                return np.eye(3) if c > 0 else Rot.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + vx + vx @ vx * (1 / (1 + c))

        R_align = rotation_aligning(ring_normal, perp_dir)

        checked = 0
        for roll_deg in range(0, 360, 10):
            R_roll = Rot.from_rotvec(np.radians(roll_deg) * view_axis).as_matrix()
            for dist in (0.2, 0.4, 0.7, 1.0, 1.3):
                R_total = R_roll @ R_align
                t_cam_ctrl = view_axis * dist
                pts_cam = (R_total @ positions.T).T + t_cam_ctrl
                normals_cam = (R_total @ normals.T).T
                in_front = pts_cam[:, 2] > 0.05
                if in_front.sum() < 4:
                    continue
                view_dir = -pts_cam / np.linalg.norm(pts_cam, axis=1, keepdims=True)
                facing = np.einsum("ij,ij->i", normals_cam, view_dir) > facing_cos
                visible = in_front & facing
                if visible.sum() < 4:
                    continue
                pts3d_vis = pts_cam[visible]
                try:
                    rvec = np.zeros(3, dtype=np.float32); tvec = np.zeros(3, dtype=np.float32)
                    px, _ = cam.project_points(pts3d_vis.astype(np.float32), rvec, tvec)
                except Exception:
                    continue
                px = np.asarray(px).reshape(-1, 2)
                if not np.isfinite(px).all():
                    continue
                in_img = (px[:, 0] > -50) & (px[:, 0] < 700) & (px[:, 1] > -50) & (px[:, 1] < 550)
                if in_img.sum() < 4:
                    continue
                px = px[in_img]
                blobs = _make_blobs(px, [20.0] * len(px))
                result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
                checked += 1
                self.assertTrue(
                    result.keep_mask.all(),
                    f"controller pose (roll={roll_deg}, dist={dist}) must never be removed",
                )
        self.assertGreater(checked, 50, "sanity: the sweep should have exercised a meaningful number of poses")


class BrightnessGuardTests(unittest.TestCase):
    def test_bright_blob_matching_lamp_geometry_is_untouched(self):
        """Same clean, straight, correctly-spaced row as CleanLampRowTests --
        geometry alone would match -- but brightness is real-LED-bright, well
        above max_brightness. The brightness guard must veto it."""
        points, _ = _lamp_row()
        bright = [200.0] * len(points)
        blobs = _make_blobs(points, bright)
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        self.assertTrue(result.keep_mask.all(), "a bright, real-LED-like blob must never be removed")
        self.assertEqual(len(result.rejected_candidates), 0)
        self.assertGreaterEqual(len(result.guarded_candidates), 1)

    def test_one_bright_blob_spares_the_whole_candidate(self):
        """A single bright blob mixed into an otherwise-dim, otherwise-clean
        lamp row must spare the ENTIRE candidate, not just that one blob --
        deliberately biased toward protecting real LEDs."""
        points, brightness = _lamp_row()
        brightness[len(points) // 2] = 200.0  # one bright outlier mid-row
        blobs = _make_blobs(points, brightness)
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        self.assertTrue(result.keep_mask.all(), "one bright blob must spare the whole candidate")

    def test_real_merged_blob_peak_needed_max_brightness_raised_to_80(self):
        """Real case (cam3, right, frame_range 1750-1800 relative frame 15):
        a genuine 23-point lamp row (residual ~1.08px, spacing 5-9px) was
        entirely spared under the OLD max_brightness=60.0 because one of its
        real split-peak points -- from a big pass-2 merged blob -- measured
        brightness=61, just 1 unit over. Uses a real lamp row shape (not
        that exact 23-point layout) with one point nudged to 61 to isolate
        the guard's own off-by-a-hair behavior."""
        points, brightness = _lamp_row(n=8)
        brightness[3] = 61.0  # one real (dim, still well under a real LED's
                                # range) peak just over the OLD 60.0 bound
        blobs = _make_blobs(points, brightness)

        result_old = detect_lamp_blobs(blobs, {**_DEFAULT_CFG, "max_brightness": 60.0})
        self.assertTrue(result_old.keep_mask.all(),
                         "with the OLD max_brightness=60.0, one point at 61 must spare "
                         "the entire otherwise-qualifying row")

        result_new = detect_lamp_blobs(blobs, {**_DEFAULT_CFG, "max_brightness": 80.0})
        self.assertFalse(result_new.keep_mask.any(),
                          "with max_brightness raised to 80.0 (this project's current "
                          "config.yml value), the same row must be fully removed")

    def test_two_real_rows_needed_max_brightness_raised_to_210(self):
        """Real case (cam3, left, frame_range 1700-1735 relative frame 24):
        two genuine, clean lamp rows (14 and 13 points, residuals
        1.20px/1.22px -- a textbook real-fixture match, confirmed visually
        against the raw frame as a continuous bright-to-dim diagonal
        gradient, not an isolated spike) were BOTH entirely spared under the
        then-current max_brightness=80.0 because each row had one member
        reading 84 and 130 respectively -- comfortably real fixture
        brightness on this recording, overlapping this project's own
        documented real accepted-LED range (median 40, up to 203). Uses two
        real lamp row shapes (not the exact 14/13-point layouts) with one
        point in each nudged to the real observed values to isolate the
        guard's behavior."""
        row_a, bright_a = _lamp_row(n=14, spacing=6.0, y=100.0, x0=100.0)
        bright_a[7] = 84.0
        row_b, bright_b = _lamp_row(n=13, spacing=6.0, y=200.0, x0=100.0)
        bright_b[6] = 130.0
        points = row_a + row_b
        brightness = bright_a + bright_b
        blobs = _make_blobs(points, brightness)

        result_old = detect_lamp_blobs(blobs, {**_DEFAULT_CFG, "max_brightness": 80.0})
        self.assertTrue(result_old.keep_mask.all(),
                         "with max_brightness=80.0, one point at 84 (row a) and one at "
                         "130 (row b) must spare BOTH otherwise-qualifying rows")

        result_new = detect_lamp_blobs(blobs, {**_DEFAULT_CFG, "max_brightness": 210.0})
        self.assertFalse(result_new.keep_mask.any(),
                          "with max_brightness raised to 210.0 (this project's current "
                          "config.yml value), both rows must be fully removed")


class MixedFrameTests(unittest.TestCase):
    def test_only_lamp_row_flagged_bright_blobs_elsewhere_untouched(self):
        lamp_points, lamp_brightness = _lamp_row(x0=100.0, y=100.0)
        bright_points = [(300.0, 300.0), (306.0, 300.0), (312.0, 300.0)]
        bright_brightness = [200.0, 200.0, 200.0]
        points = lamp_points + bright_points
        brightness = lamp_brightness + bright_brightness
        blobs = _make_blobs(points, brightness)
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)

        n_lamp = len(lamp_points)
        self.assertFalse(result.keep_mask[:n_lamp].any(), "the dim lamp row should be removed")
        self.assertTrue(result.keep_mask[n_lamp:].all(), "the bright unrelated blobs must be untouched")


class DegenerateInputTests(unittest.TestCase):
    def test_empty_blob_result(self):
        blobs = BlobResult.empty()
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        self.assertEqual(len(result.keep_mask), 0)
        self.assertEqual(result.rejected_candidates, [])

    def test_single_blob(self):
        blobs = _make_blobs([(50.0, 50.0)], [20.0])
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        self.assertTrue(result.keep_mask.all())

    def test_fewer_than_min_points_blobs(self):
        points, brightness = _lamp_row(n=2)
        blobs = _make_blobs(points, brightness)
        result = detect_lamp_blobs(blobs, _DEFAULT_CFG)
        self.assertTrue(result.keep_mask.all())


class DimContextRestrictionTests(unittest.TestCase):
    """Tests for restrict_dim_context_to_kept_neighborhood (src/blob_detector.py's
    caller uses this to shrink the "dim" pool fed into detect_lamp_blobs to
    only points actually near a kept detection -- see that function's own
    docstring for the real performance blowup this fixes: 8 kept vs 429
    scattered dim blobs on one real frame once min_threshold was lowered)."""

    def _dim(self, points):
        centroids = [(float(x), float(y)) for x, y in points]
        contours = [np.zeros((3, 2), dtype=np.float32) for _ in points]  # content irrelevant here
        brightnesses = [10.0 for _ in points]
        return centroids, contours, brightnesses

    def test_dim_point_near_a_kept_blob_survives(self):
        centroids, contours, brightnesses = self._dim([(105.0, 100.0)])  # 5px from the kept blob
        kept = np.array([[100.0, 100.0]])
        out_c, out_cnt, out_b = restrict_dim_context_to_kept_neighborhood(
            centroids, contours, brightnesses, kept, radius_px=150.0)
        self.assertEqual(out_c, centroids)
        self.assertEqual(len(out_cnt), 1)
        self.assertEqual(out_b, brightnesses)

    def test_dim_point_far_from_every_kept_blob_is_dropped(self):
        centroids, contours, brightnesses = self._dim([(1000.0, 1000.0)])  # far away
        kept = np.array([[100.0, 100.0]])
        out_c, out_cnt, out_b = restrict_dim_context_to_kept_neighborhood(
            centroids, contours, brightnesses, kept, radius_px=150.0)
        self.assertEqual(out_c, [])
        self.assertEqual(out_cnt, [])
        self.assertEqual(out_b, [])

    def test_mixed_population_only_near_points_survive_in_original_order(self):
        pts = [(105.0, 100.0), (1000.0, 1000.0), (100.0, 240.0), (-500.0, 0.0)]
        centroids, contours, brightnesses = self._dim(pts)
        kept = np.array([[100.0, 100.0]])
        out_c, out_cnt, out_b = restrict_dim_context_to_kept_neighborhood(
            centroids, contours, brightnesses, kept, radius_px=150.0)
        self.assertEqual(out_c, [pts[0], pts[2]])

    def test_multiple_kept_anchors_union_of_neighborhoods(self):
        # Near the SECOND kept blob only -- must still survive (any kept
        # anchor being close enough is sufficient, not just the first).
        centroids, contours, brightnesses = self._dim([(505.0, 500.0)])
        kept = np.array([[100.0, 100.0], [500.0, 500.0]])
        out_c, _, _ = restrict_dim_context_to_kept_neighborhood(
            centroids, contours, brightnesses, kept, radius_px=150.0)
        self.assertEqual(out_c, centroids)

    def test_no_kept_anchor_leaves_dim_points_untouched(self):
        """No kept blob at all -- an anchor-less dim point's relevance can't
        be decided by this rule, so existing (unrestricted) behavior is
        preserved rather than dropping everything."""
        centroids, contours, brightnesses = self._dim([(1000.0, 1000.0)])
        out_c, out_cnt, out_b = restrict_dim_context_to_kept_neighborhood(
            centroids, contours, brightnesses, np.empty((0, 2)), radius_px=150.0)
        self.assertEqual(out_c, centroids)
        self.assertEqual(out_b, brightnesses)

    def test_no_dim_points_is_a_no_op(self):
        out_c, out_cnt, out_b = restrict_dim_context_to_kept_neighborhood(
            [], [], [], np.array([[0.0, 0.0]]), radius_px=150.0)
        self.assertEqual(out_c, [])


class MaxPoolSizeCircuitBreakerTests(unittest.TestCase):
    """Tests for detect_lamp_blobs' own max_pool_size cap (a defense-in-depth
    safety net independent of the dim-context restriction above -- e.g. a
    scene with many densely-clustered KEPT, not just dim, blobs)."""

    def test_pool_under_cap_runs_normally(self):
        points, brightness = _lamp_row(n=10)
        blobs = _make_blobs(points, brightness)
        result = detect_lamp_blobs(blobs, {**_DEFAULT_CFG, "max_pool_size": 50})
        self.assertEqual(result.skipped_pool_too_large, 0)
        self.assertFalse(result.keep_mask.any(), "still removes the real lamp row under the cap")

    def test_pool_over_cap_is_skipped_not_crashed(self):
        points, brightness = _lamp_row(n=10)
        blobs = _make_blobs(points, brightness)
        result = detect_lamp_blobs(blobs, {**_DEFAULT_CFG, "max_pool_size": 5})
        self.assertEqual(result.skipped_pool_too_large, 10)
        self.assertTrue(result.keep_mask.all(), "skip is the safe failure mode -- nothing removed, not a crash")

    def test_max_pool_size_zero_means_unbounded(self):
        points, brightness = _lamp_row(n=10)
        blobs = _make_blobs(points, brightness)
        result = detect_lamp_blobs(blobs, {**_DEFAULT_CFG, "max_pool_size": 0})
        self.assertEqual(result.skipped_pool_too_large, 0)
        self.assertFalse(result.keep_mask.any())


class PerClusterMaxPoolSizeTests(unittest.TestCase):
    """Regression test for a real failure found 2026-09-10 on a real cam0
    frame: the frame's total candidate pool was 299 points, but 237 of those
    were one unrelated dense blob elsewhere in the frame (nowhere near
    lamp-row spacing -- e.g. a busy textured surface). The OLD max_pool_size
    check summed the WHOLE frame's pool (299 > 150) and skipped the line
    search entirely, silently leaving a completely clean, isolated real lamp
    row in the frame's opposite corner un-recognized. max_pool_size must be
    scoped to each spatially-connected cluster (see detect_lamp_blobs' own
    comment), not the whole frame, so a small isolated real fixture is never
    held hostage by unrelated noise volume elsewhere in the same frame."""

    def test_isolated_real_row_found_despite_unrelated_oversized_cluster(self):
        lamp_points, lamp_brightness = _lamp_row(n=10, spacing=7.0, x0=100.0, y=100.0, brightness=20.0)
        # An unrelated dense cluster of 200 points, tightly packed (well
        # under spacing_min_px so it forms ONE connected component), far
        # away from the real lamp row -- mimics a real busy/textured surface
        # that has nothing to do with any lamp fixture.
        rng = np.random.default_rng(1)
        noise_cluster = rng.uniform(0.0, 5.0, size=(200, 2)) + np.array([1000.0, 1000.0])
        noise_brightness = rng.uniform(10.0, 55.0, size=200)

        all_points = lamp_points + [tuple(p) for p in noise_cluster]
        all_brightness = lamp_brightness + list(noise_brightness)
        blobs = _make_blobs(all_points, all_brightness)

        cfg = {**_DEFAULT_CFG, "max_pool_size": 150}
        result = detect_lamp_blobs(blobs, cfg)

        n_lamp = len(lamp_points)
        self.assertFalse(result.keep_mask[:n_lamp].any(),
                          "the isolated real lamp row must still be found and removed, "
                          "regardless of an unrelated oversized cluster elsewhere in the frame")
        self.assertTrue(result.keep_mask[n_lamp:].all(),
                         "the oversized unrelated cluster must be skipped (safe failure mode), not touched")
        self.assertEqual(result.skipped_pool_too_large, 200,
                          "only the oversized cluster's own point count should be reported skipped, "
                          "not the whole frame's pool")


class PerfRegressionTests(unittest.TestCase):
    """Guards against the real-world perf blowup found 2026-09-09: a lowered
    min_threshold flooding detect_lamp_blobs with hundreds of scattered dim
    points (429 dim vs 8 kept on one real cam3 frame), driving
    _find_dominant_lines' roughly-cubic-in-pool-size cost (see its own
    docstring) from sub-millisecond to multiple seconds. This test builds a
    large SYNTHETIC pool with the same shape (a real lamp row plus a lot of
    scattered noise) directly (not the real recording's captured data, which
    isn't part of this repo) and asserts detect_lamp_blobs still finds and
    removes the genuine row, well within a generous time budget."""

    def test_large_noisy_pool_stays_fast_and_still_finds_the_real_row(self):
        rng = np.random.default_rng(0)
        lamp_points, lamp_brightness = _lamp_row(n=15, spacing=7.0, x0=100.0, y=100.0, brightness=20.0)
        # 400 scattered noise points spread across a much larger area, well
        # outside the real row's own footprint -- mimics the real frame's
        # dim-noise population (this test exercises detect_lamp_blobs
        # directly, i.e. WITHOUT the caller-side dim-context restriction in
        # src/blob_detector.py, so it's a genuine stress test of
        # _find_dominant_lines' own batched-seed rewrite, not of the
        # restriction that normally keeps this pool small in production).
        noise_points = rng.uniform(0.0, 2000.0, size=(400, 2))
        noise_brightness = rng.uniform(10.0, 55.0, size=400)

        all_points = lamp_points + [tuple(p) for p in noise_points]
        all_brightness = lamp_brightness + list(noise_brightness)
        blobs = _make_blobs(all_points, all_brightness)

        # max_pool_size=0 (unbounded) -- this test exercises the batched
        # _find_dominant_lines rewrite itself, not the separate max_pool_size
        # circuit breaker (which would otherwise skip a 415-point pool
        # outright under its own default of 150, short-circuiting before
        # ever reaching the code this test is actually meant to stress).
        t0 = time.time()
        result = detect_lamp_blobs(blobs, {**_DEFAULT_CFG, "max_pool_size": 0})
        elapsed = time.time() - t0

        n_lamp = len(lamp_points)
        self.assertFalse(result.keep_mask[:n_lamp].any(), "the real lamp row must still be found and removed")
        self.assertLess(elapsed, 5.0,
                         f"took {elapsed:.2f}s on a 415-point pool -- the batched rewrite should keep "
                         f"this well under a second even with 400 noise points; regression if this fails")


if __name__ == "__main__":
    unittest.main()
