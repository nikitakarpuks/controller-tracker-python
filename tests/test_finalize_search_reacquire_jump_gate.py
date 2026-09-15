"""Regression coverage for CameraTracker.finalize_search's tight pose-jump
guard (src/controller.py) during re-acquisition, fixed 2026-09-10.

Background: finalize_search runs a tight, per-axis pose-jump check against
BOTH self.prev_pose and the IMU-extrapolated predicted_pose, rejecting a
candidate implausibly far from either (see that method's own module
docstring for the 2026-09-06 case motivating the OR-logic between the two).
Before this fix, the WHOLE check was skipped whenever _reacquiring was True
(self.tracking_lost_last_frame set, regardless of whether prev_pose itself
was still populated -- see CameraTracker.__init__'s own comment on why
tracking_lost_last_frame, not prev_pose being None, is what "reacquiring"
actually means) -- even when predicted_pose was fully available and valid.
Since predicted_pose is already velocity/time-aware, unlike prev_pose's
fixed per-frame delta, there's no reason a recent loss should also disable
checking against it. Found investigating a real case: a reacquisition frame
landed with a ~38deg rotation jump vs the IMU state at dt=11.1ms (~3400deg/s
-- not physically possible) and sailed through with no per-camera check at
all, deferring entirely to the much looser fusion-level
implausible_jump_rot_deg gate (60deg flat, not time-scaled).

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_finalize_search_reacquire_jump_gate
"""
import unittest
from collections import deque

import cv2
import numpy as np

from src.controller import CameraTracker
from src.transformations import Transform


def _rvec_deg(axis, deg):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return (axis * np.radians(deg)).astype(np.float32).reshape(3, 1)


def _make_tracker(prev_pose, tracking_lost_last_frame, matching_cfg=None,
                   last_good_pose=None, last_good_pose_ts_ns=None,
                   last_good_pose_quality=None, pose_history_len=3):
    """CameraTracker.__init__ builds a real PoseSearcher/Camera stack this
    test doesn't need -- finalize_search only reads plain self attributes,
    so bypass __init__ and set exactly those, mirroring the approach
    tests/test_controller_imu_only_propagation.py uses for ControllerTracker.

    last_good_pose_quality defaults to None (no recorded quality -- matches
    a fresh CameraTracker's own __init__ default), which correctly disables
    the 2026-09-12 quality-comparison rescue for every existing test here
    that doesn't pass it explicitly.

    pose_history_len defaults to 3 -- MATURE (see pose_history_window's own
    config comment: 1=constant pos, 2=const vel, 3+=linear fit), which
    correctly disables the 2026-09-12 immature-velocity strong-match floor
    for every existing test here that doesn't pass it explicitly (none of
    them are testing THAT mechanism -- see NoVelocityProximityStrongMatchFloorTests
    for the tests that are, which pass 1 or 2 explicitly). Contents are
    dummy placeholders -- these tests pass predicted_pose directly rather
    than having finalize_search derive it from pose_history, so only the
    LENGTH matters here, not what's actually in it."""
    tracker = CameraTracker.__new__(CameraTracker)
    tracker.camera = type("C", (), {"camera_idx": 0})()
    tracker.model = type("M", (), {"name": "right_controller"})()
    tracker.T_world_cam = Transform(np.eye(3), np.zeros(3))
    tracker.prev_pose = prev_pose
    tracker.prev_prev_pose = None
    tracker.tracking_lost_last_frame = tracking_lost_last_frame
    tracker.vel_ema = None
    tracker.pose_history = deque(
        [(_ZERO_RVEC, _ZERO_TVEC, 0)] * pose_history_len, maxlen=5)
    tracker.last_good_pose = last_good_pose
    tracker.last_good_pose_ts_ns = last_good_pose_ts_ns
    tracker.last_good_pose_quality = last_good_pose_quality
    tracker.consecutive_failures = 0
    tracker._matching_cfg = matching_cfg or {}
    return tracker


def _solution(rvec, tvec, error=0.1):
    """assignment defaults to 6 pairs (clears strong_match_inliers' own
    default of 6, alongside error=0.1 clearing strong_match_error_px=0.5) --
    this file's tests are about the pos/rot jump-gate logic, not about the
    2026-09-12 strong-match floor for a vel_ema-is-None proximity accept
    (see LastGoodPoseQualityRescueTests / finalize_search's own comment on
    that floor), so the default candidate here should clear it trivially
    unless a test deliberately overrides assignment/error to test otherwise."""
    return {
        "rvec": np.asarray(rvec, dtype=np.float32).reshape(3, 1),
        "tvec": np.asarray(tvec, dtype=np.float32).reshape(3, 1),
        "error": error,
        "assignment": [(i, i) for i in range(6)],
        "method": "proximity",
    }


_ZERO_RVEC = np.zeros((3, 1), dtype=np.float32)
_ZERO_TVEC = np.zeros((3, 1), dtype=np.float32)


class ReacquisitionPredictedPoseGateTests(unittest.TestCase):
    def test_implausible_jump_vs_predicted_pose_now_rejected_while_reacquiring(self):
        """The core regression: previously this whole check was skipped once
        tracking_lost_last_frame was set, no matter how implausible the
        candidate was against a perfectly good predicted_pose."""
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),   # stale, but still populated
            tracking_lost_last_frame=True,        # -> _reacquiring=True
        )
        predicted_pose = (_ZERO_RVEC, _ZERO_TVEC)
        # Same position as predicted_pose (clears the pos check trivially),
        # but a 90deg rotation jump -- default scalar max_angle_deg=25.0.
        candidate = _solution(_rvec_deg([0, 0, 1], 90.0), _ZERO_TVEC)

        result = tracker.finalize_search(
            candidate, predicted_pose, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNone(result, "a 90deg jump vs predicted_pose must be rejected "
                                   "even while re-acquiring")

    def test_plausible_candidate_vs_predicted_pose_still_accepted_while_reacquiring(self):
        """Converse of the above -- must not become MORE strict than before:
        a genuinely plausible candidate (small delta vs predicted_pose) is
        still accepted during re-acquisition."""
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=True,
        )
        predicted_pose = (_ZERO_RVEC, _ZERO_TVEC)
        # 2deg is comfortably under even the tightest default vs-predicted_pose
        # per-axis threshold (pose_jump_pred_rot_thresh_base_deg z=4.5deg @
        # speed=0, since vel_ema is None here -> speed_est_m_s=0).
        candidate = _solution(_rvec_deg([0, 0, 1], 2.0), _ZERO_TVEC)

        result = tracker.finalize_search(
            candidate, predicted_pose, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNotNone(result, "a small, plausible delta vs predicted_pose "
                                      "must still be accepted while re-acquiring")

    def test_no_predicted_pose_and_reacquiring_skips_the_gate_entirely(self):
        """Neither reference is usable (no predicted_pose, and vs-prev is
        excluded by _reacquiring) -- must fall through unrejected by THIS
        gate (matches pre-fix behavior for this specific combination; the
        looser fusion-level / last_good_pose-based re-acquisition checks
        elsewhere are unaffected by this change)."""
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=True,
        )
        candidate = _solution(_rvec_deg([0, 0, 1], 90.0), _ZERO_TVEC)

        result = tracker.finalize_search(
            candidate, None, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNotNone(result, "with no predicted_pose available, re-acquiring "
                                      "must not be gated by this check at all")


class LastGoodPoseRescueDuringReacquisitionTests(unittest.TestCase):
    """Regression for a bug found 2026-09-10 investigating a user question
    about accumulated IMU drift across a multi-frame loss: the tight guard's
    predicted_pose-is-authoritative branch was applying UNCONDITIONALLY
    whenever predicted_pose existed, even when the SEPARATE cold-start /
    re-acquisition block above had already found the candidate plausible
    against last_good_pose. During an active loss, predicted_pose is built
    from the fusion filter's raw IMU-only dead-reckoning (a different, less-
    validated error regime than the vision-history-based extrapolation the
    tight guard's own thresholds were empirically fit against -- see that
    block's own comment), so it must not be able to silently override an
    already-accepted last_good_pose-based decision."""

    def test_predicted_pose_disagreement_does_not_override_last_good_pose_agreement(self):
        tracker = _make_tracker(
            prev_pose=None,
            tracking_lost_last_frame=True,   # -> _reacquiring=True
            last_good_pose=(_ZERO_RVEC, _ZERO_TVEC),
            last_good_pose_ts_ns=0,
        )
        # Candidate agrees closely with last_good_pose (0deg) -- the
        # cold-start block's own OR-check against it passes comfortably.
        candidate = _solution(_rvec_deg([0, 0, 1], 1.0), _ZERO_TVEC)
        # But predicted_pose (raw IMU-only dead-reckoning during the loss)
        # says 90deg -- wildly disagrees, well past the tight vs-predicted
        # threshold.
        predicted_pose = (_rvec_deg([0, 0, 1], 90.0), _ZERO_TVEC)

        result = tracker.finalize_search(
            candidate, predicted_pose, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNotNone(result, "last_good_pose agreement must still accept during "
                                      "re-acquisition even when predicted_pose disagrees -- "
                                      "predicted_pose must not independently re-veto a "
                                      "decision the cold-start block already made")

    def test_predicted_pose_and_last_good_pose_both_disagreeing_still_rejects(self):
        tracker = _make_tracker(
            prev_pose=None,
            tracking_lost_last_frame=True,
            last_good_pose=(_ZERO_RVEC, _ZERO_TVEC),
            last_good_pose_ts_ns=0,
        )
        candidate = _solution(_rvec_deg([0, 0, 1], 90.0), _ZERO_TVEC)
        # predicted_pose at 0deg -- agrees with last_good_pose, NOT with the
        # candidate, so both references genuinely disagree with candidate.
        predicted_pose = (_ZERO_RVEC, _ZERO_TVEC)

        result = tracker.finalize_search(
            candidate, predicted_pose, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNone(result, "implausible vs both last_good_pose and predicted_pose "
                                   "must still be rejected during re-acquisition")


class ImmatureVelocityProximityStrongMatchFloorTests(unittest.TestCase):
    """Regression for a real case found 2026-09-12 (frame_range 3850-3950
    relative frame 51): pose_history was at n=2 (self.vel_ema a single raw,
    UNSMOOTHED step derived from the PRIOR frame's own weak 5-inlier/0.98px
    accept -- see apply()'s own vel_ema computation), so proximity search's
    per-LED candidate neighborhoods (built from that immature prediction)
    degraded badly: 9 of 13 visible LEDs' neighborhoods found zero candidates
    at all, the primary top-k search hit its own 8000-node safety cap without
    finding a result, and the eventual fallback matched only 4/13 LEDs at up
    to 2.07px residual -- yet nothing rejected it, because the position half
    of the jump gate is deliberately disabled/loosened at this depth (see
    NoVelocityEstimatePositionCheckTests) and rotation alone happened to
    agree. An n=1 case (self.vel_ema is None outright -- no velocity at all,
    not just an unsmoothed one) has the exact same failure shape and is
    covered here too. This floor requires a proximity accept to independently
    clear strong_match_inliers/strong_match_error_px on its own merits
    whenever pose_history is below pose_history_window's "3+=linear fit"
    maturity tier, since there's no reliable prediction-based check to lean
    on instead in that window."""

    def test_weak_proximity_match_at_n1_is_rejected(self):
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,   # continuous tracking, just past bootstrap
            pose_history_len=1,
        )
        # Real case's own numbers: 4 inliers, 0.52px -- below strong_match_inliers=6.
        candidate = _solution(_ZERO_RVEC, _ZERO_TVEC, error=0.52)
        candidate["assignment"] = [(i, i) for i in range(4)]

        result = tracker.finalize_search(
            candidate, (_ZERO_RVEC, _ZERO_TVEC), blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNone(result, "a weak (below strong_match_inliers/error_px) proximity "
                                   "accept at n=1 must be rejected")

    def test_weak_proximity_match_at_n2_is_rejected(self):
        """The real frame-51 case: n=2, self.vel_ema a single raw
        (unsmoothed) step -- not None, but just as immature as n=1."""
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,
            pose_history_len=2,
        )
        candidate = _solution(_ZERO_RVEC, _ZERO_TVEC, error=0.52)
        candidate["assignment"] = [(i, i) for i in range(4)]

        result = tracker.finalize_search(
            candidate, (_ZERO_RVEC, _ZERO_TVEC), blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNone(result, "a weak proximity accept at n=2 (single-step, unsmoothed "
                                   "vel_ema) must be rejected exactly like n=1")

    def test_strong_proximity_match_at_n1_is_accepted(self):
        """Converse: a genuinely strong proximity match in the same immature
        window must NOT be penalized just for lacking a mature velocity
        estimate -- this floor only rejects candidates that are ALSO weak on
        their own merits."""
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,
            pose_history_len=1,
        )
        candidate = _solution(_ZERO_RVEC, _ZERO_TVEC, error=0.1)  # default 6 inliers, clears the floor

        result = tracker.finalize_search(
            candidate, (_ZERO_RVEC, _ZERO_TVEC), blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNotNone(result, "a strong proximity match must still be accepted even "
                                      "with an immature velocity estimate")

    def test_weak_proximity_match_at_mature_n3_is_unaffected(self):
        """Once pose_history reaches the "3+=linear fit" maturity tier, this
        floor must not apply at all -- a weak match there is judged only by
        the normal jump-gate logic, unaffected by this addition."""
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,
            pose_history_len=3,
        )
        candidate = _solution(_ZERO_RVEC, _ZERO_TVEC, error=0.52)
        candidate["assignment"] = [(i, i) for i in range(4)]

        result = tracker.finalize_search(
            candidate, (_ZERO_RVEC, _ZERO_TVEC), blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNotNone(result, "a weak match at mature pose_history depth (n>=3) must "
                                      "not be rejected by the immature-velocity floor")

    def test_weak_brute_force_match_at_n1_is_exempt(self):
        """The floor is proximity-only -- a weak brute-force/cold result
        (method != 'proximity') must NOT be rejected by this check, since
        cold search's own coverage-fallback policy ('take the best available
        candidate') is an intentional design choice with no cheaper fallback
        to defer to."""
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,
            pose_history_len=1,
        )
        candidate = _solution(_ZERO_RVEC, _ZERO_TVEC, error=0.52)
        candidate["assignment"] = [(i, i) for i in range(4)]
        candidate["method"] = "p3p_systematic"

        result = tracker.finalize_search(
            candidate, (_ZERO_RVEC, _ZERO_TVEC), blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNotNone(result, "a weak brute-force/cold result must not be rejected "
                                      "by the proximity-only strong-match floor")


class LastGoodPoseQualityRescueTests(unittest.TestCase):
    """Regression for a real case found 2026-09-12 (frame_range 3850-3950
    relative frame 50): a clean 11-inlier/0.12px re-acquisition candidate was
    rejected for disagreeing by 2.6deg on one rotation axis against a
    last_good_pose that was itself only a 5-inlier/0.98px COVERAGE-FALLBACK
    accept one frame earlier -- the weak reference's own inaccuracy was the
    far more likely explanation, not ~3600deg/s of real rotation. See
    last_good_pose_quality's own comment (src/controller.py, CameraTracker
    __init__) for the full real-case numbers."""

    def test_candidate_clearly_beating_a_weak_reference_is_rescued(self):
        tracker = _make_tracker(
            prev_pose=None,
            tracking_lost_last_frame=True,
            last_good_pose=(_ZERO_RVEC, _ZERO_TVEC),
            last_good_pose_ts_ns=0,
            last_good_pose_quality=(5, 0.98),  # weak: below strong_match_inliers/error_px defaults (6/0.5)
        )
        # 90deg disagreement -- well past the tight vs-predicted/last_good_pose
        # thresholds on its own -- but the candidate clears strong_match_inliers/
        # error_px outright (12 >= 6, 0.10 <= 0.5) AND beats the reference by
        # more than 2x inliers / less than half the error.
        candidate = _solution(_rvec_deg([0, 0, 1], 90.0), _ZERO_TVEC, error=0.10)
        candidate["assignment"] = [(i, i) for i in range(12)]
        predicted_pose = None  # no predicted-pose rescue available -- quality rescue must stand alone

        result = tracker.finalize_search(
            candidate, predicted_pose, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNotNone(result, "a candidate clearing strong_match_inliers/error_px AND "
                                      "beating a weak last_good_pose by 2x inliers/half error "
                                      "must be rescued, not rejected as an implausible jump")

    def test_candidate_only_marginally_better_than_weak_reference_still_rejects(self):
        """Same weak reference, but the candidate only edges it out slightly
        (6 inliers vs 5, not >= 2x) -- must NOT be rescued; a small
        improvement isn't strong enough evidence the reference was wrong."""
        tracker = _make_tracker(
            prev_pose=None,
            tracking_lost_last_frame=True,
            last_good_pose=(_ZERO_RVEC, _ZERO_TVEC),
            last_good_pose_ts_ns=0,
            last_good_pose_quality=(5, 0.98),
        )
        candidate = _solution(_rvec_deg([0, 0, 1], 90.0), _ZERO_TVEC, error=0.45)
        candidate["assignment"] = [(i, i) for i in range(6)]
        predicted_pose = None

        result = tracker.finalize_search(
            candidate, predicted_pose, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNone(result, "a merely-somewhat-better candidate (not >= 2x inliers / "
                                   "<= half error) must still be rejected -- the quality "
                                   "rescue is deliberately conservative")

    def test_candidate_beating_an_already_strong_reference_is_not_rescued(self):
        """The reference itself was already strong (matches/beats
        strong_match_inliers/error_px) -- there's no "weak reference" story
        to rescue against, so the implausible jump must still reject even
        though the candidate numerically beats it."""
        tracker = _make_tracker(
            prev_pose=None,
            tracking_lost_last_frame=True,
            last_good_pose=(_ZERO_RVEC, _ZERO_TVEC),
            last_good_pose_ts_ns=0,
            last_good_pose_quality=(9, 0.15),  # already strong
        )
        candidate = _solution(_rvec_deg([0, 0, 1], 90.0), _ZERO_TVEC, error=0.05)
        candidate["assignment"] = [(i, i) for i in range(20)]
        predicted_pose = None

        result = tracker.finalize_search(
            candidate, predicted_pose, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNone(result, "a reference that was already strong gives no basis for "
                                   "the quality rescue, regardless of how good the candidate is")


class IndependentVetoDuringContinuousTrackingTests(unittest.TestCase):
    """Regression for the second half of the 2026-09-10 fix: predicted_pose
    is now the AUTHORITATIVE reference whenever it's available, not just a
    rescue consulted after vs-prev_pose already failed. A candidate that
    agrees with prev_pose but disagrees with the (empirically tighter)
    predicted_pose must now be rejected -- this is exactly the real case
    that motivated this fix: a per-camera candidate that individually passed
    the loose vs-prev_pose check on both cameras, but disagreed with
    predicted_pose on the same axis on both, sailed through undetected
    before this change (the fusion level then saw a ~38deg disagreement vs
    the IMU state with nothing having rejected it upstream)."""

    def test_predicted_pose_veto_even_when_prev_pose_agrees(self):
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,   # continuous tracking
        )
        # 20deg vs prev_pose -- comfortably under the default 25deg scalar
        # vs-prev_pose threshold (vs-prev_pose alone would ACCEPT this).
        candidate = _solution(_rvec_deg([0, 0, 1], 20.0), _ZERO_TVEC)
        # But predicted_pose says only 13deg -- a 7deg delta on z, over the
        # default vs-predicted_pose z-threshold (4.5deg @ speed=0).
        predicted_pose = (_rvec_deg([0, 0, 1], 13.0), _ZERO_TVEC)

        result = tracker.finalize_search(
            candidate, predicted_pose, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNone(result, "predicted_pose disagreement must independently reject "
                                   "even when vs-prev_pose agrees")

    def test_predicted_pose_agreement_still_accepts(self):
        """Converse -- must not become trigger-happy: candidate close to
        BOTH prev_pose and predicted_pose is still accepted."""
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,
        )
        candidate = _solution(_rvec_deg([0, 0, 1], 3.0), _ZERO_TVEC)
        predicted_pose = (_rvec_deg([0, 0, 1], 2.0), _ZERO_TVEC)

        result = tracker.finalize_search(
            candidate, predicted_pose, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNotNone(result, "agreement with both references must still accept")


class ContinuousTrackingUnaffectedTests(unittest.TestCase):
    """Confirms the pre-existing continuous-tracking (_reacquiring=False)
    OR-rescue behavior (a candidate implausible vs prev_pose ALONE is still
    rescued by predicted_pose agreement) is unaffected by this fix -- the
    exact 2026-09-06 case from this method's own module docstring."""

    def test_rescued_by_predicted_pose_exactly_as_before(self):
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,   # continuous tracking
        )
        # 30deg vs prev_pose (over the default 25deg scalar threshold)...
        predicted_pose = (_rvec_deg([0, 0, 1], 28.0), _ZERO_TVEC)
        candidate = _solution(_rvec_deg([0, 0, 1], 30.0), _ZERO_TVEC)
        # ...but only ~2deg vs predicted_pose -- comfortably under even the
        # tightest default vs-predicted_pose per-axis threshold (z=4.5deg @
        # speed=0, since vel_ema is None here -> speed_est_m_s=0).

        result = tracker.finalize_search(
            candidate, predicted_pose, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNotNone(result, "a candidate implausible vs prev_pose alone must "
                                      "still be rescued by a plausible predicted_pose "
                                      "during continuous tracking")

    def test_implausible_vs_both_still_rejected(self):
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,
        )
        predicted_pose = (_ZERO_RVEC, _ZERO_TVEC)
        candidate = _solution(_rvec_deg([0, 0, 1], 90.0), _ZERO_TVEC)

        result = tracker.finalize_search(
            candidate, predicted_pose, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNone(result, "implausible vs both prev_pose and predicted_pose "
                                   "must still be rejected during continuous tracking")


class NoVelocityEstimatePositionCheckTests(unittest.TestCase):
    """Regression for the 2026-09-11 fix: at n=1 (self.vel_ema is None --
    true exactly when pose_history has fewer than 2 accepted frames),
    _predict_pose's translation half returns pose_history[0]'s own tvec
    completely unchanged (see its own n==1 docstring branch) -- the SAME
    reference point vs_prev_pose already checks, just against a threshold
    far tighter than vs_prev_pose's own (pose_jump_pred_pos_thresh_base_mm
    default 61.5mm vs. vs_prev_pose's default 150mm scalar / production's
    180/180/200mm per-axis). Before this fix, a candidate plausible vs
    prev_pose could still get independently vetoed by predicted_pose on
    position ALONE, purely because no velocity estimate existed yet to
    widen that threshold -- confirmed on a real recording (frame_range
    1000-1010, relative frame 4): a candidate 90mm from prev_pose (well
    inside prev_pose's own 150mm/180mm bound) was checked against
    predicted_pose's 61.5mm floor and failed on position, on top of also
    failing on rotation that same frame. Rotation is untouched by this fix
    (it's gyro-measured, independent of vision-history depth) -- only the
    position component is disabled when there's no established velocity."""

    def test_position_only_disagreement_not_rejected_without_velocity_estimate(self):
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,   # continuous tracking, just past bootstrap
        )
        self.assertIsNone(tracker.vel_ema, "test assumes no velocity estimate yet (n=1)")
        # 90mm from prev_pose -- comfortably under prev_pose's own default
        # 150mm scalar threshold (vs_prev_pose alone would ACCEPT this).
        candidate_tvec = np.array([0.0, 0.0, 0.09], dtype=np.float32).reshape(3, 1)
        candidate = _solution(_ZERO_RVEC, candidate_tvec)
        # predicted_pose's tvec is pose_history[0]'s own tvec, unchanged (n==1's
        # real behavior) -- same reference point as prev_pose, zero rotation
        # disagreement, so only the position check could possibly reject.
        predicted_pose = (_ZERO_RVEC, _ZERO_TVEC)

        result = tracker.finalize_search(
            candidate, predicted_pose, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNotNone(result, "a position-only disagreement vs predicted_pose "
                                      "must not independently reject when there is no "
                                      "established velocity estimate -- vs_prev_pose's "
                                      "own (looser) judgment already covers this case")

    def test_rotation_disagreement_still_rejected_without_velocity_estimate(self):
        """Converse -- the position-check disable must not silence rotation
        checking too: a real rotation disagreement (gyro-measured, valid
        regardless of vision-history depth) still independently rejects,
        even with a large position agreement (tvec identical) alongside it."""
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,
        )
        self.assertIsNone(tracker.vel_ema)
        candidate = _solution(_rvec_deg([0, 1, 0], 24.0), _ZERO_TVEC)
        predicted_pose = (_ZERO_RVEC, _ZERO_TVEC)  # 24deg y-axis disagreement, default thresh 12.1deg

        result = tracker.finalize_search(
            candidate, predicted_pose, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNone(result, "rotation disagreement vs predicted_pose must still "
                                   "reject even with no established velocity estimate")


class TimeScaledPrevPoseJumpGateTests(unittest.TestCase):
    """Regression for 2026-09-13 (full jump-check audit, user-directed): vs-
    prev_pose used to be a FLAT per-axis budget (pose_jump_pos_thresh_m/
    _rot_thresh_deg) regardless of how much real time had actually elapsed
    since prev_pose was captured -- a frame silently skipped without ever
    entering "lost" state got the exact same tight budget as a normal
    back-to-back frame. Now base + max_plausible_hand_speed_m_s/
    _ang_speed_deg_s * dt_s (pose_jump_pos_thresh_base_m/_rot_thresh_base_deg
    plus the shared rate), dt_s = real elapsed time from prev_pose's own
    pose_history[0][2] timestamp (0 in this fixture, see _make_tracker) to
    frame_ts_ns.

    Also SCALAR (Euclidean distance / total rotation angle), not per-axis --
    a second, same-day fix (also user-directed): independent per-axis
    rejection let diagonal motion hide arbitrary extra real distance (a
    candidate offset (563,36,31)mm, magnitude ~566mm, got REJECTED while one
    offset (440,440,440)mm, magnitude ~762mm -- MORE actual displacement --
    would have been ACCEPTED). See pose_jump_pos_thresh_base_m's own config
    comment for the full account."""

    _CFG = {
        "pose_jump_pos_thresh_base_m": 0.21,
        "pose_jump_rot_thresh_base_deg": 27.0,
        "max_plausible_hand_speed_m_s": 3.0,
        "max_plausible_hand_ang_speed_deg_s": 2000.0,
    }

    def test_jump_within_widened_budget_at_large_dt_is_accepted(self):
        """0.5m on x is only reachable once the budget has widened past
        prev_pose's own base (0.21m) -- at dt=200ms the budget is
        0.21 + 3.0*0.2 = 0.81m, comfortably covering it."""
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,
            matching_cfg=self._CFG,
        )
        candidate = _solution(_ZERO_RVEC, np.array([0.5, 0.0, 0.0], dtype=np.float32))

        result = tracker.finalize_search(
            candidate, None, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=200_000_000,
        )

        self.assertIsNotNone(result, "a jump within the dt-widened budget must be accepted")

    def test_same_jump_rejected_at_near_zero_dt(self):
        """Identical 0.5m candidate, but frame_ts_ns close to prev_pose's own
        ts=0 -- budget stays near base-only (~0.21m), well under 0.5m."""
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,
            matching_cfg=self._CFG,
        )
        candidate = _solution(_ZERO_RVEC, np.array([0.5, 0.0, 0.0], dtype=np.float32))

        result = tracker.finalize_search(
            candidate, None, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

        self.assertIsNone(result, "the same 0.5m jump must be rejected at near-zero dt "
                                   "(base-only budget ~0.21m) -- proves this is real "
                                   "time-scaling, not just a looser flat constant")

    def test_rotation_axis_also_time_scaled(self):
        """Same widening applies to rotation, not just position -- 90deg on
        one axis is only reachable once dt has widened the budget past the
        27deg base (at dt=200ms: 27.0 + 2000*0.2 = 427deg, comfortably
        covering it; near dt=0 it would reject, same shape as the position
        pair above)."""
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,
            matching_cfg=self._CFG,
        )
        candidate = _solution(_rvec_deg([0, 0, 1], 90.0), _ZERO_TVEC)

        result = tracker.finalize_search(
            candidate, None, blobs=np.zeros((0, 2), dtype=np.float32),
            allow_expensive_fallback=False, frame_ts_ns=200_000_000,
        )

        self.assertIsNotNone(result, "a rotation jump within the dt-widened budget must be accepted")


class ScalarNotPerAxisPositionJumpGateTests(unittest.TestCase):
    """Regression for the exact asymmetry the user found and asked to fix
    (2026-09-13, same day): a per-axis check rejects a concentrated
    (single-axis) displacement while ACCEPTING a diagonal one of LARGER
    total real distance, purely because the diagonal one happens to be
    spread across axes -- e.g. a real case (563,36,31)mm (magnitude ~566mm)
    got rejected, while (440,440,440)mm (magnitude ~762mm, MORE actual
    displacement) would have passed. A scalar (Euclidean-distance) check
    can't produce that asymmetry: two candidates with the SAME real 3D
    displacement magnitude, however differently distributed across x/y/z,
    always get the SAME verdict."""

    _CFG = {
        "pose_jump_pos_thresh_base_m": 0.21,
        "pose_jump_rot_thresh_base_deg": 27.0,
        "max_plausible_hand_speed_m_s": 3.0,
        "max_plausible_hand_ang_speed_deg_s": 2000.0,
    }

    def _result_for(self, tvec):
        tracker = _make_tracker(
            prev_pose=(_ZERO_RVEC, _ZERO_TVEC),
            tracking_lost_last_frame=False,
            matching_cfg=self._CFG,
        )
        candidate = _solution(_ZERO_RVEC, np.array(tvec, dtype=np.float32))
        return tracker.finalize_search(
            candidate, None, blobs=np.zeros((0, 2), dtype=np.float32),
            # near-zero dt -> base-only budget (~0.21m)
            allow_expensive_fallback=False, frame_ts_ns=1_000_000,
        )

    def test_same_magnitude_concentrated_and_diagonal_get_the_same_verdict(self):
        magnitude = 0.30  # comfortably over the ~0.21m base-only budget
        concentrated = self._result_for([magnitude, 0.0, 0.0])
        diagonal = self._result_for([magnitude / 3 ** 0.5] * 3)  # same Euclidean norm, spread across 3 axes

        self.assertIsNone(concentrated, "a 0.30m single-axis jump must be rejected")
        self.assertIsNone(diagonal, "a diagonal jump of the SAME 0.30m total magnitude must ALSO be "
                                     "rejected -- a per-axis check would have passed this despite "
                                     "identical real displacement, purely because it's spread across axes")

    def test_a_smaller_diagonal_magnitude_is_accepted_same_as_smaller_concentrated(self):
        """Converse -- must not become MORE strict than before either: a
        genuinely small displacement is accepted regardless of how it's
        distributed across axes."""
        magnitude = 0.15  # comfortably under the ~0.21m base-only budget
        concentrated = self._result_for([magnitude, 0.0, 0.0])
        diagonal = self._result_for([magnitude / 3 ** 0.5] * 3)

        self.assertIsNotNone(concentrated, "a small single-axis jump must be accepted")
        self.assertIsNotNone(diagonal, "the same small magnitude, spread across axes, must also be accepted")


if __name__ == "__main__":
    unittest.main()
