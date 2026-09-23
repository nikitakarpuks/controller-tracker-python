"""Tests for src/static_light_map.py -- Phase 2 of the mocap-based
static-light (ceiling-lamp) exclusion feature: the persisted map format and
the voxel-vote accumulator. Ray casting itself is exercised in
tests/test_static_light_geometry.py; these tests feed the accumulator
directly-constructed (origin, direction) pairs so a bug in one module can't
be misattributed to the other.

Note on synthetic point choice: true_point below is deliberately placed at
the MIDDLE of a voxel (fractional coordinate .5 in voxel_size_m=0.1 units),
not on a round number -- a point sitting exactly on a voxel grid line is a
worst-case pathological input for a coarse voxel grid (floating-point noise
alone can flip which side of the boundary a given ray's discretized sample
lands on), and isn't representative of a real lamp position.

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_static_light_map
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.static_light_map import (
    StaticLightMap, StaticLightVoxelAccumulator,
    save_static_light_map, load_static_light_map,
)

_TRUE_POINT = np.array([2.05, 0.95, 2.45])

# Headset origins spread ~1-1.5m apart (plausible over a few seconds of real
# head motion) around a point ~3.3m away -- comfortably clears both the
# 0.15m positional-spread and 15deg angular-spread defaults.
_SPREAD_ORIGINS = [
    [0.0, 0.0, 0.0], [0.9, 0.3, 0.15], [-0.6, 0.75, -0.3],
    [0.3, -0.9, 0.6], [-0.75, -0.45, 0.45], [1.05, 0.0, -0.6],
]


def _rays_to_point(point, origins):
    """(origin, direction) pairs pointing exactly at `point` from each of `origins`."""
    out = []
    for o in origins:
        o = np.asarray(o, dtype=np.float64)
        d = point - o
        out.append((o, d / np.linalg.norm(d)))
    return out


class TestSaveLoadRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        m = StaticLightMap(
            voxel_size_m=0.1,
            voxel_room_positions=np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 2.5]]),
            voxel_radii_m=np.array([0.12, 0.2]),
            vote_counts=np.array([42, 55]),
            camera_calib_id="abc123",
            created_from_recordings=["rec1", "rec2"],
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "map.json"
            save_static_light_map(m, path)
            self.assertTrue(path.exists())
            loaded = load_static_light_map(path)

        np.testing.assert_allclose(loaded.voxel_room_positions, m.voxel_room_positions)
        np.testing.assert_allclose(loaded.voxel_radii_m, m.voxel_radii_m)
        np.testing.assert_array_equal(loaded.vote_counts, m.vote_counts)
        self.assertEqual(loaded.camera_calib_id, m.camera_calib_id)
        self.assertEqual(loaded.created_from_recordings, m.created_from_recordings)
        self.assertEqual(loaded.voxel_size_m, m.voxel_size_m)

    def test_round_trip_empty_map(self):
        m = StaticLightMap(
            voxel_size_m=0.1,
            voxel_room_positions=np.zeros((0, 3)),
            voxel_radii_m=np.zeros((0,)),
            vote_counts=np.zeros((0,), dtype=np.int64),
            camera_calib_id="abc123",
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty_map.json"
            save_static_light_map(m, path)
            loaded = load_static_light_map(path)
        self.assertEqual(loaded.voxel_room_positions.shape, (0, 3))
        self.assertEqual(loaded.created_from_recordings, [])


class TestVoxelAccumulatorStaticVsMoving(unittest.TestCase):
    """The core discriminative behavior the whole feature depends on: a
    genuinely static point gets confirmed, a moving one (or a degenerate
    near-stationary-headset case) does not."""

    def _accumulator(self):
        return StaticLightVoxelAccumulator(voxel_size_m=0.1, min_votes=30,
                                            min_positional_spread_m=0.15,
                                            min_angular_spread_deg=15.0)

    def test_static_point_is_confirmed_within_voxel_size(self):
        acc = self._accumulator()
        for origin, direction in _rays_to_point(_TRUE_POINT, _SPREAD_ORIGINS * 6):
            acc.add_observation(origin, direction, max_range_m=6.0, step_m=0.02)

        result = acc.finalize(camera_calib_id="test-rig", created_from_recordings=["synthetic"])

        self.assertEqual(len(result.voxel_room_positions), 1)
        dist = np.linalg.norm(result.voxel_room_positions[0] - _TRUE_POINT)
        self.assertLess(dist, acc.voxel_size_m)
        self.assertGreaterEqual(result.vote_counts[0], 30)
        self.assertEqual(result.camera_calib_id, "test-rig")
        self.assertEqual(result.created_from_recordings, ["synthetic"])

    def test_moving_point_is_not_confirmed(self):
        """Each observation points at a DIFFERENT nearby location (simulating
        a drifting controller) -- votes never stack up in one voxel the way
        a genuinely fixed point's would, even with the same number of
        observations and the same headset-motion spread as the static case."""
        acc = self._accumulator()
        rng = np.random.default_rng(0)
        for origin in _SPREAD_ORIGINS * 6:
            moving_point = _TRUE_POINT + rng.uniform(-0.4, 0.4, size=3)
            (o, d), = _rays_to_point(moving_point, [origin])
            acc.add_observation(o, d, max_range_m=6.0, step_m=0.02)

        result = acc.finalize()
        self.assertEqual(len(result.voxel_room_positions), 0)

    def test_degenerate_zero_spread_is_rejected_despite_high_vote_count(self):
        """A perfectly stationary headset staring at a fixed point racks up
        plenty of votes but with zero positional/angular spread -- must not
        be confirmed, since this is indistinguishable from a controller
        resting still relative to a non-moving headset."""
        acc = self._accumulator()
        origins = [[0.0, 0.0, 0.0]] * 50
        for origin, direction in _rays_to_point(_TRUE_POINT, origins):
            acc.add_observation(origin, direction, max_range_m=6.0, step_m=0.02)

        result = acc.finalize()
        self.assertEqual(len(result.voxel_room_positions), 0)


class TestFinalizeClusterMerging(unittest.TestCase):
    """Hand-constructed vote data (bypassing ray casting entirely) to test
    the voxel-merge/radius-derivation logic in isolation."""

    def _confirmable_origins(self):
        # Same spread as _SPREAD_ORIGINS, reused directly as pre-populated
        # voxel entries -- known to clear min_positional_spread_m=0.1 and
        # min_angular_spread_deg=5.0 against a voxel a few meters away.
        return [np.array(o) for o in _SPREAD_ORIGINS * 3]

    def test_adjacent_confirmed_voxels_merge_into_one_light(self):
        acc = StaticLightVoxelAccumulator(voxel_size_m=0.1, min_votes=10,
                                           min_positional_spread_m=0.1,
                                           min_angular_spread_deg=5.0)
        origins = self._confirmable_origins()
        acc._voxels[(20, 10, 24)] = {"count": len(origins), "origins": list(origins)}
        acc._voxels[(21, 10, 24)] = {"count": len(origins), "origins": list(origins)}

        result = acc.finalize()

        self.assertEqual(len(result.voxel_room_positions), 1)
        self.assertEqual(result.vote_counts[0], 2 * len(origins))

    def test_far_apart_confirmed_voxels_stay_separate(self):
        acc = StaticLightVoxelAccumulator(voxel_size_m=0.1, min_votes=10,
                                           min_positional_spread_m=0.1,
                                           min_angular_spread_deg=5.0)
        origins = self._confirmable_origins()
        acc._voxels[(0, 0, 0)] = {"count": len(origins), "origins": list(origins)}
        acc._voxels[(50, 0, 0)] = {"count": len(origins), "origins": list(origins)}

        result = acc.finalize()

        self.assertEqual(len(result.voxel_room_positions), 2)
        np.testing.assert_array_equal(np.sort(result.vote_counts), [len(origins), len(origins)])

    def test_non_adjacent_voxels_within_merge_distance_still_merge(self):
        """Two confirmed voxels 2 grid-steps apart (0.2m at voxel_size=0.1)
        are NOT 26-adjacent (adjacency only ever spans 1 step, max diagonal
        sqrt(3)*0.1=0.173m), but are well within the default distance-based
        cluster_merge_distance_m (0.235m) -- this is the real gap this
        distance-based merge was added for: two confirmed voxels for the
        SAME physical fixture (real recording, two bulbs of one lamp) landed
        0.2236m apart, which no grid-adjacency test could ever bridge."""
        acc = StaticLightVoxelAccumulator(voxel_size_m=0.1, min_votes=10,
                                           min_positional_spread_m=0.1,
                                           min_angular_spread_deg=5.0)
        origins = self._confirmable_origins()
        acc._voxels[(20, 10, 24)] = {"count": len(origins), "origins": list(origins)}
        acc._voxels[(20, 10, 26)] = {"count": len(origins), "origins": list(origins)}

        result = acc.finalize()

        self.assertEqual(len(result.voxel_room_positions), 1)
        self.assertEqual(result.vote_counts[0], 2 * len(origins))

    def test_voxels_just_beyond_merge_distance_stay_separate(self):
        """3 grid-steps apart (0.3m) exceeds the default cluster_merge_
        distance_m (0.235m at voxel_size=0.1) -- must NOT merge, or the
        distance-based merge would eventually conflate genuinely distinct
        nearby fixtures instead of just bridging one fixture's own
        multi-voxel spread."""
        acc = StaticLightVoxelAccumulator(voxel_size_m=0.1, min_votes=10,
                                           min_positional_spread_m=0.1,
                                           min_angular_spread_deg=5.0)
        origins = self._confirmable_origins()
        acc._voxels[(20, 10, 24)] = {"count": len(origins), "origins": list(origins)}
        acc._voxels[(20, 10, 27)] = {"count": len(origins), "origins": list(origins)}

        result = acc.finalize()

        self.assertEqual(len(result.voxel_room_positions), 2)
        np.testing.assert_array_equal(np.sort(result.vote_counts), [len(origins), len(origins)])


if __name__ == "__main__":
    unittest.main()
