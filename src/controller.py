import copy
import csv
import os
import time
import cv2
import numpy as np
from collections import deque
from loguru import logger
from typing import List, Tuple, Optional, Dict, Set, FrozenSet, Union

from src._pnp import _project_points
from src._visibility import _visible_mask, _cross_occluded_mask
from src.camera import Camera
from src.debug_config import is_continuous_sequence
from src.transformations import Transform
from src._self_calibration import SelfCalibrator
from src.imu_data import integrate_gyro_segment
from src.mocap_data import world_pose
from src.pose_fusion import PoseFusionFilter
from src.pose_fusion_heuristic import HeuristicPoseFusionFilter


# =========================================================
# Temporary diagnostic: predicted-vs-actual jump-tolerance stats
# =========================================================
# Opt-in via CONTROLLER_TRACKER_JUMP_STATS_CSV=/path/to/out.csv -- a no-op
# (zero overhead, no file touched) otherwise. One row per solved per-camera
# candidate in finalize_search, BEFORE either jump gate runs, so this
# captures the real predicted-vs-actual distribution the gates are meant to
# bound -- used to empirically retune pose_jump_pos_thresh_m/_rot_thresh_xyz_deg
# (2026-09-06 investigation) rather than guessing. Not wired through any
# config file / call-site signature on purpose -- this is a one-off analysis
# tool, not a permanent feature; remove this whole block once done, or leave
# it (env-gated, so harmless either way) if it turns out useful to rerun
# later.
_JUMP_STATS_CSV_PATH = os.environ.get("CONTROLLER_TRACKER_JUMP_STATS_CSV")
_jump_stats_writer = None
_jump_stats_file = None


def _log_jump_stats_row(**fields) -> None:
    global _jump_stats_writer, _jump_stats_file
    if not _JUMP_STATS_CSV_PATH:
        return
    if _jump_stats_writer is None:
        _jump_stats_file = open(_JUMP_STATS_CSV_PATH, "w", newline="")
        _jump_stats_writer = csv.DictWriter(_jump_stats_file, fieldnames=list(fields.keys()))
        _jump_stats_writer.writeheader()
    _jump_stats_writer.writerow(fields)


# =========================================================
# 1. DATA STRUCTURES
# =========================================================

class ControllerLED:
    def __init__(self, position: np.ndarray, normal: np.ndarray):
        self.position = np.asarray(position, dtype=np.float32).reshape(3)
        self.normal = np.asarray(normal, dtype=np.float32).reshape(3)


class ControllerModel:
    def __init__(self, leds: List[ControllerLED], name: str):
        self.name = name
        self.leds = leds

        # Precompute for speed
        self.positions = np.stack([l.position for l in leds])
        self.normals = np.stack([l.normal for l in leds])


def create_leds_from_config(cfg) -> List[ControllerLED]:
    leds_cfg = cfg["CalibrationInformation"]["ControllerLeds"]

    return [
        ControllerLED(
            position=np.array(led["Position"], dtype=np.float32),
            normal=np.array(led["Normal"], dtype=np.float32),
        )
        for led in leds_cfg
    ]


# =========================================================
# 2. GEOMETRY — fit frustum + bake handle primitives
# =========================================================

def mirror_primitives(prim_cfg: dict) -> dict:
    """Return a YZ-plane (X-reflected) copy of a handle_primitives config block."""
    p = copy.deepcopy(prim_cfg)
    for b in p.get("boxes", []):
        b["center"][0] = -b["center"][0]
        if "axes" in b:
            for row in b["axes"]:
                row[0] = -row[0]
    for cy in p.get("cylinders", []):
        cy["center"][0] = -cy["center"][0]
        cy["angle"] = -cy.get("angle", 0.0)
    return p


def _build_other_cameras_blobs(cameras: Dict[int, Camera], obs_src: Dict[int, np.ndarray],
                                exclude_cid: int) -> List[Tuple]:
    """Other-camera blob data for brute_search_tier's cross-camera coverage check
    (PoseSearcher.new_brute_state's other_cameras_blobs param): every camera besides
    exclude_cid that has real blob data this frame. Not filtered by any per-camera
    persistence/confirm-frames gate -- that gate only decides whether to attempt a
    brute search FROM a camera, not whether its raw blobs are valid aux evidence for
    scoring another camera's candidates."""
    return [
        (cameras[ocid], obs_src[ocid], None)
        for ocid in obs_src
        if ocid != exclude_cid and obs_src[ocid] is not None and len(obs_src[ocid]) > 0
    ]


def _weak_solo_accept_cids(cam_solutions: List[dict], eligible_cids: List[int],
                            av_count_by_cid: Dict[int, int], matching_cfg: dict) -> List[int]:
    """Shared by ControllerTracker.update and TrackingSystem.update_warm_batch:
    both accept a controller's cheap-search result the instant ANY camera
    produces something, even a single low-confidence camera while another
    camera that had real detected blobs this frame came up with nothing --
    see either call site's own comment for the real recording (frame_range
    800-900 relative frames 50/51/53) that motivated this.

    Returns the eligible-but-unsolved camera ids (empty = nothing to flag)
    when EVERY accepted cam_solutions entry is "weak": either its own inlier
    count is below matching.strong_match_inliers (the SAME floor
    PoseSearcher's own brute strong_found early-exit uses), OR its
    inliers/available-blobs ratio is below matching.
    weak_solo_blob_utilization_floor (default 0.7, empirically set -- see
    that config key's own comment). Empty whenever every camera that had
    blobs this frame already contributed (nothing left to cross-check
    against) or at least one accepted solution is already strong."""
    if not cam_solutions or len(cam_solutions) >= len(eligible_cids):
        return []
    strong_floor = int(matching_cfg.get('strong_match_inliers', 6))
    util_floor   = float(matching_cfg.get('weak_solo_blob_utilization_floor', 0.7))

    def _is_weak(cs: dict) -> bool:
        n_inliers = len(cs["solution"]["assignment"])
        if n_inliers < strong_floor:
            return True
        n_av = av_count_by_cid.get(cs["cam_id"], 0)
        return n_av > 0 and (n_inliers / n_av) < util_floor

    if not all(_is_weak(cs) for cs in cam_solutions):
        return []
    solved_cids = {cs["cam_id"] for cs in cam_solutions}
    return [cid for cid in eligible_cids if cid not in solved_cids]


# =========================================================
# 3. TRACKER (per camera + controller)
# =========================================================

def _gyro_rel_R_for(gyro_data: Optional[tuple], pose_history, frame_ts_ns: int):
    """gyro_data: (t_gyro, gyro_body) for one controller, already calibrated,
    axis-corrected into body frame, and clock-offset-corrected into the vision
    timestamp domain (see main.py) -- or None if no IMU data was loaded for this
    controller/run. Returns the (3,3) rel-rotation integrate_gyro_segment
    computes over [pose_history[0]'s timestamp, frame_ts_ns], or None (no gyro
    data, no pose history yet, or frame_ts_ns outside the gyro's covered range)."""
    if gyro_data is None or not pose_history:
        return None
    ts0 = int(pose_history[0][2])
    return integrate_gyro_segment(gyro_data[0], gyro_data[1], ts0, frame_ts_ns)


def _predicted_world_for_vel_ema(fusion_filter, trackers, frame_ts_ns: int):
    """self._fusion_filter.predict(frame_ts_ns) -- called AT MOST ONCE per
    controller per frame (not once per camera): the result depends only on
    frame_ts_ns and the fusion filter's own internal state, not on any one
    camera's pose_history, so every camera of the same controller would
    otherwise redundantly repeat the exact same accel+gyro integration work
    (see predict()'s own cost) for what's mathematically an identical
    world-frame answer. Also skips calling predict() at all when no camera
    actually needs the fallback this frame (the common, steady-state case --
    see _vel_ema_with_imu_fallback's own gate), so this adds zero overhead
    once vision-only tracking is running normally.

    Returns None (skip predict() entirely) if there's no fusion filter, no
    camera-tracker currently in the narrow gap _imu_vel_ema_override exists
    for, or the fusion filter has no established velocity of its own yet
    (fusion_filter.velocity_established -- see its own comment) -- this
    entire mechanism exists to derive a POSITION velocity substitute from the
    fusion filter's dead-reckoning, so if the fusion filter itself has never
    measured a real velocity (still on a fabricated v=0 from a bootstrap/
    reset/fail-open), there is nothing here to derive; otherwise returns
    predict()'s own (R_pred_world, p_pred_world) or None result unchanged."""
    if fusion_filter is None:
        return None
    if not any(t.vel_ema is None and len(t.pose_history) <= 2 for t in trackers):
        return None
    if not fusion_filter.velocity_established:
        return None
    return fusion_filter.predict(frame_ts_ns)


def _imu_vel_ema_override(predicted_world, camera: "Camera", pose_history, frame_ts_ns: int):
    """Camera-frame position-per-second rate derived from the fusion filter's
    own accel+gyro world-frame dead-reckoning (HeuristicPoseFusionFilter/
    PoseFusionFilter.predict(), already validated elsewhere in this project --
    see those classes' own docstrings), for use as _predict_pose's
    vel_ema_rate override in the ONE narrow gap where the pure-vision fallback
    isn't available yet: right after a fresh (re)acquisition, before enough
    real accepted frames exist for a trustworthy vision-only velocity
    estimate. Callers (see _vel_ema_with_imu_fallback below) only ever reach
    this when tracker.vel_ema is already None -- this NEVER overrides an
    already-populated vision-only estimate (this project's own "trust vision
    over IMU" hierarchy: IMU only fills a gap vision genuinely can't cover
    yet, never competes with a real vision-derived one).

    Found 2026-09-09 investigating a real recording: after a full tracking-
    loss reacquisition, pose_history's first 1-2 entries alone gave
    _predict_pose's own n=1 (constant-position) / n=2 (fractional-velocity-
    from-2-noisy-points) branches a translation prediction bad enough to fail
    the proximity search's tight neighbourhood radius on the next TWO frames
    running, forcing two extra expensive cold brute-force (p3p_systematic)
    re-detects before enough real vision samples (n=3) accumulated for the
    existing linear-fit branch to average the noise out on its own. The
    fusion filter's own dead-reckoning already solves exactly this (real
    accel+gyro integration, not a 2-point vision-only slope) -- reused here
    rather than inventing a second, separate accel-integration path.

    predicted_world: fusion_filter.predict(frame_ts_ns)'s own return value,
    computed ONCE per controller per frame by the caller (see
    _predicted_world_for_vel_ema) -- NOT re-computed here, since multiple
    cameras of the same controller share the identical world-frame answer.

    Returns None (caller keeps vel_ema=None, i.e. today's unchanged behavior)
    if there's no usable prediction (no fusion filter, no coverage, g_world
    not converged, no bootstrapped tracking state -- same fail-open contract
    as predict() everywhere else) or no pose_history yet."""
    if predicted_world is None or not pose_history:
        return None
    R_pred_world, p_pred_world = predicted_world
    T_cam_ctrl_pred = camera.T_world_cam.inverse().compose(Transform(R_pred_world, p_pred_world))
    ts0 = int(pose_history[0][2])
    dt_s = (frame_ts_ns - ts0) / 1e9
    if dt_s <= 0:
        return None
    tvec0 = np.asarray(pose_history[0][1], np.float64).reshape(3)
    return ((T_cam_ctrl_pred.t.astype(np.float64) - tvec0) / dt_s).astype(np.float32)


def _vel_ema_with_imu_fallback(tracker: "CameraTracker", predicted_world, camera: "Camera", frame_ts_ns: int):
    """tracker.vel_ema if it's already populated OR pose_history is long
    enough (>2) that _predict_pose's own linear-fit branch already averages
    out vision-only noise on its own (see that method's docstring) --
    otherwise falls back to _imu_vel_ema_override for the specific narrow gap
    where neither is true yet (see that function's own docstring).

    predicted_world: see _predicted_world_for_vel_ema -- computed once per
    controller per frame by the caller, not here."""
    if tracker.vel_ema is not None or len(tracker.pose_history) > 2:
        return tracker.vel_ema
    return _imu_vel_ema_override(predicted_world, camera, tracker.pose_history, frame_ts_ns)


# Stage 3 gravity-alignment diagnostic (thaytan's OpenHMD dev-diary technique):
# log-only for now, see the plan notes on why this isn't wired to reject anything
# yet -- not confident enough in the false-positive rate on this data to let it
# force brute-force retries on a shared, critical path.
_GRAVITY_LOW_DYNAMICS_TOL_MS2 = 2.0   # |accel| within this of 9.81 m/s^2 => trust it as a gravity reference
_GRAVITY_DISAGREEMENT_DEG     = 20.0  # log a warning above this angle between two consecutive low-dynamics readings


def _interp_imu_sample(t: np.ndarray, data: np.ndarray, ts_ns: int) -> Optional[np.ndarray]:
    """Per-axis linear interpolation of one (t, data) IMU stream at ts_ns, or None if
    ts_ns falls outside [t[0], t[-1]] (no real coverage to interpolate from) -- shared
    by _log_gravity_consistency and PoseFusionFilter's LiveGravityEstimator feed
    (_commit_fused_solution), both of which need exactly this one-instant sample."""
    if ts_ns < t[0] or ts_ns > t[-1]:
        return None
    return np.array([np.interp(ts_ns, t, data[:, i]) for i in range(3)])


def _log_gravity_consistency(ctrl_name: str, R_now: np.ndarray, ts_now: int,
                              R_prev: Optional[np.ndarray], ts_prev: Optional[int],
                              accel_data: Optional[tuple],
                              accel_now: Optional[np.ndarray] = None) -> None:
    """Compares the accelerometer-implied 'down' direction (rotated into the
    controller's fused frame via R_now) against the same quantity computed at
    the previous accepted frame (R_prev) -- both gated on the accelerometer
    reading being close to 9.81 m/s^2 at that instant (i.e. probably not
    contaminated by real linear acceleration). A large disagreement between two
    low-dynamics readings means at least one of the two orientations is
    probably wrong. Doesn't need a true world-frame reference: the camera rig
    only drifts slowly relative to gravity frame-to-frame, so "previous
    accepted frame" is a good enough proxy over one ~16ms gap.

    accel_now: caller's already-interpolated accel sample at ts_now, if it happened
    to need one anyway (see _commit_fused_solution's gravity-estimator feed) -- avoids
    interpolating the same stream at the same instant twice. None recomputes it here."""
    if accel_data is None or R_prev is None or ts_prev is None:
        return
    t_accel, accel_body = accel_data
    a_now  = accel_now if accel_now is not None else _interp_imu_sample(t_accel, accel_body, ts_now)
    a_prev = _interp_imu_sample(t_accel, accel_body, ts_prev)
    if a_now is None or a_prev is None:
        return
    if abs(np.linalg.norm(a_now) - 9.81) > _GRAVITY_LOW_DYNAMICS_TOL_MS2:
        return
    if abs(np.linalg.norm(a_prev) - 9.81) > _GRAVITY_LOW_DYNAMICS_TOL_MS2:
        return

    g_now  = R_now  @ (a_now  / np.linalg.norm(a_now))
    g_prev = R_prev @ (a_prev / np.linalg.norm(a_prev))
    cos_angle = np.clip(float(g_now @ g_prev), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    if angle_deg > _GRAVITY_DISAGREEMENT_DEG:
        logger.bind(cat="matching_decisions").warning(
            f"[{ctrl_name}] gravity-direction check: {angle_deg:.1f}° disagreement between "
            f"consecutive low-dynamics accel readings (diagnostic only, not rejecting)")


def cheap_search_core(
    pose_searcher,
    prior: dict,
    matching_cfg: dict,
    blobs: np.ndarray,
    frame_ts_ns: int,
    blob_radii: Optional[np.ndarray] = None,
    blob_brightnesses: Optional[np.ndarray] = None,
    other_cameras_blobs: Optional[List] = None,
    blob_mask: Optional[np.ndarray] = None,
    occluders_per_cam: Optional[Dict] = None,
) -> Tuple[Optional[Dict], Optional[Tuple[np.ndarray, np.ndarray]], Optional[Tuple], Optional[Tuple]]:
    """
    Pure proximity + prior_constrained search — no brute-force, no CameraTracker
    instance needed, just a PoseSearcher and an explicit prior-state bundle. This is
    what lets a worker process run the cheap search using only its own resident
    PoseSearcher (see src/parallel_search.py) — a live CameraTracker's mutable
    tracking state can't be shared with a worker (fork only gives it a stale
    snapshot), so the caller must pass that state in explicitly instead.

    prior: {'prev_pose', 'prev_prev_pose', 'pose_history', 'vel_ema', 'prev_assignment',
    'gyro_rel_R'} — the same fields CameraTracker.search_cheap() reads from self.
    'pose_history' entries are (rvec, tvec, ts_ns); 'vel_ema' is a position-per-second
    rate, not a raw step; 'gyro_rel_R' is an optional (3,3) body-frame relative
    rotation from measured gyro (see _gyro_rel_R_for) that overrides rotation
    prediction — see CameraTracker._predict_pose's docstring.

    frame_ts_ns: the current frame's real capture timestamp (nanoseconds, parsed
    from the frame's filename in main.py) — _predict_pose extrapolates against
    this exact elapsed time rather than assuming uniform frame spacing.

    Returns (solution_or_None, predicted_pose, normalized_prev_pose,
    normalized_prev_prev_pose) — the normalized values mirror the idempotent
    (3,1)/(3,) canonicalisation CameraTracker.search_cheap() applies to
    self.prev_pose/self.prev_prev_pose as a side effect; a caller with live state
    (CameraTracker.search_cheap) writes these back, a stateless worker just returns
    them for the orchestrator to write back into the real tracker.
    """
    blobs   = np.asarray(blobs, dtype=np.float32).reshape(-1, 2)
    n_blobs = len(blobs)

    if blob_mask is not None:
        _avail_idx  = np.where(blob_mask)[0].astype(np.int32)
        blobs_prox  = blobs[_avail_idx]
        brts_prox   = blob_brightnesses[_avail_idx] if blob_brightnesses is not None else None
        n_available = len(blobs_prox)
    else:
        blobs_prox  = blobs
        brts_prox   = blob_brightnesses
        n_available = n_blobs

    _cfg = matching_cfg
    _use_proximity = bool(_cfg.get('use_proximity_match', True))

    prev_pose       = prior.get('prev_pose')
    prev_prev_pose  = prior.get('prev_prev_pose')
    pose_history    = prior.get('pose_history')
    vel_ema         = prior.get('vel_ema')
    prev_assignment = prior.get('prev_assignment')
    gyro_rel_R      = prior.get('gyro_rel_R')

    # Normalise prev_pose shapes (idempotent — canonicalises (3,1) rvec and (3,) tvec)
    if prev_pose is not None:
        rvec, tvec = prev_pose
        prev_pose = (
            np.asarray(rvec, dtype=np.float32).reshape(3, 1),
            np.asarray(tvec, dtype=np.float32).reshape(3),
        )
    if prev_prev_pose is not None:
        rvec, tvec = prev_prev_pose
        prev_prev_pose = (
            np.asarray(rvec, dtype=np.float32).reshape(3, 1),
            np.asarray(tvec, dtype=np.float32).reshape(3),
        )

    predicted_pose = CameraTracker._predict_pose(
        pose_history,
        frame_ts_ns,
        weight_decay=float(_cfg.get("pose_prediction_weight_decay", 0.7)),
        vel_ema_rate=vel_ema,
        gyro_rel_R=gyro_rel_R,
    )

    # Velocity-scaled search gates: expand proximity radius proportionally to
    # SPEED (m/s) -- NOT to how far the point-prediction moved this particular
    # frame. BUG (found investigating an oscillating neighbourhood-size report):
    # this used to be ‖predicted.t - prev.t‖, which equals |vel_ema| * dt_target
    # once vel_ema is active -- i.e. displacement over the ACTUAL upcoming gap.
    # On a recording whose own frame rate alternates (e.g. 30/60fps), dt_target
    # itself swings ~2x frame to frame, so the SAME real hand speed produced a
    # halved margin right after a 60fps-cadence step and a doubled one right
    # after a 30fps-cadence step, purely from which cadence happened to apply
    # that frame -- nothing to do with the controller's actual motion. vel_ema
    # is already a genuine, dt-normalised m/s rate (see
    # ControllerTracker._propagate_pose_history), so converting it to a pixel
    # margin needs a FIXED reference period, not this frame's own variable
    # dt_target, or the normalisation vel_ema already did gets undone right
    # back here.
    _v_px = 0.0
    if vel_ema is not None and prev_pose is not None:
        _depth  = max(float(np.asarray(prev_pose[1]).reshape(3)[2]), 0.1)
        _ref_dt = float(_cfg.get('proximity_expansion_velocity_ref_dt_s', 0.0333))
        _v_px   = float(np.linalg.norm(vel_ema)) * _ref_dt * pose_searcher.camera.fx / _depth
    _base_expansion = float(_cfg.get('proximity_expansion_px', 8.0))
    _prox_vel_k     = float(_cfg.get('proximity_expansion_velocity_k', 0.0))
    _eff_expansion  = _base_expansion + _prox_vel_k * _v_px
    # Uncertainty term: larger neighbourhood when prediction history is short.
    # Decays as 1/n — full boost at n=1 (constant-position), half at n=2, etc.
    _uncertainty_k = float(_cfg.get('proximity_expansion_uncertainty_k', 0.0))
    if _uncertainty_k > 0.0:
        _eff_expansion += _uncertainty_k / max(len(pose_history), 1)
    # Depth term: closer controller → larger pixel-space uncertainty → bigger neighbourhood.
    _depth_k = float(_cfg.get('proximity_expansion_depth_k', 0.0))
    if _depth_k > 0.0 and predicted_pose is not None:
        _ctrl_depth = max(float(predicted_pose[1].reshape(3)[2]), 0.01)
        _eff_expansion += _depth_k / _ctrl_depth

    solution = None

    if prev_pose is not None:
        # --- Primary: proximity (fast, assignment-locked) ---
        if _use_proximity and n_available >= 3:
            solution = pose_searcher.proximity_search(
                blobs_prox, predicted_pose,
                blob_brightnesses=brts_prox,
                other_cameras_blobs=other_cameras_blobs,
                occluders_per_cam=occluders_per_cam,
                expansion_px=_eff_expansion,
            )
            # Remap proximity result indices from filtered → full array space.
            if solution is not None and blob_mask is not None:
                solution['assignment'] = [(_avail_idx[b], lid) for b, lid in solution['assignment']]
                solution['_orig_idx']  = True

        # ------------------------------------------------------------------
        # Low-blob-count fallback: prior-constrained translation solve
        # P2P (3 blobs): fix R, solve t from 2 pairs, validate with 3rd.
        # P1P (2 blobs): fix R + depth, solve (tx,ty) from 1 pair, validate with 2nd.
        # ------------------------------------------------------------------
        if (solution is None and prev_assignment is not None
                and 2 <= n_available <= 3):
            logger.bind(cat="matching_decisions").debug(f"[{pose_searcher._ctrl} | cam {pose_searcher._cam} | track] n_blobs={n_available} + prior → prior_constrained_match")
            solution = pose_searcher.constrained_search(
                blobs_prox, predicted_pose,
                prior_assignment=prev_assignment,
                other_cameras_blobs=other_cameras_blobs,
            )
            if solution is not None and blob_mask is not None:
                solution['assignment'] = [(_avail_idx[b], lid) for b, lid in solution['assignment']]
                solution['_orig_idx']  = True

    return solution, predicted_pose, prev_pose, prev_prev_pose


class CameraTracker:
    def __init__(self, camera: Camera, model: ControllerModel, matching_cfg: dict = None, geometry_cfg: dict = None):
        self.camera = camera
        self.model = model

        # T_world_cam: camera frame → world/IMU frame (used to express solutions in world frame)
        self.T_world_cam: Transform = camera.T_world_cam

        # Tracking state — current frame
        self.prev_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.prev_prev_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None  # for velocity extrapolation
        self.prev_assignment = None

        # Last frame where tracking was confirmed good.
        # Retained across loss events so brute re-acquisition can be
        # validated for plausibility (pose-jump guard).
        self.last_good_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        # Real timestamp last_good_pose was captured at -- set alongside it,
        # always (see the one live setter in ControllerTracker._commit_fused_
        # solution). Needed by finalize_search's cold-start staleness widening
        # to use REAL elapsed time, not a frame-count proxy -- this
        # recording's own frame rate oscillates (~11-22ms), so consecutive_
        # failures * an assumed nominal frame period diverges from the true
        # elapsed gap over a multi-frame loss (found auditing every
        # prediction/staleness site in this pipeline for exactly this class
        # of bug -- this was the one still using a count-based proxy).
        self.last_good_pose_ts_ns: Optional[int] = None
        # (n_inliers, error_px) this camera's own solve reached when
        # last_good_pose was captured -- set alongside it (ControllerTracker.
        # apply() and _commit_fused_solution's per-camera loop). Lets
        # finalize_search's re-acquisition gate tell "this reference itself
        # was a weak/marginal accept" apart from "this reference was a clean,
        # confident one" -- both currently get the exact same fixed jump
        # threshold, which is wrong: a brand-new candidate that clearly beats
        # a weak reference on both inliers and error is evidence the
        # REFERENCE was inaccurate, not that the candidate is implausible.
        # Confirmed on a real recording (frame_range 3850-3950 relative frame
        # 50): a clean 11-inlier/0.12px re-acquisition candidate was rejected
        # for disagreeing by just 2.6deg over threshold on one rotation axis
        # (40.5deg vs a 37.9deg widened threshold) against a last_good_pose
        # that was itself only a 5-inlier/0.98px COVERAGE-FALLBACK accept one
        # frame earlier -- the far more likely explanation is that the WEAK
        # reference's own orientation estimate was off by that much, not that
        # the controller physically rotated ~40deg in 11ms (>3600deg/s).
        self.last_good_pose_quality: Optional[Tuple[int, float]] = None
        self.last_good_assignment = None

        # Consecutive frames without a valid solution.
        self.consecutive_failures: int = 0

        # Consecutive frames (cold-start only) this camera has shown >= min_inliers
        # available blobs — see the cold brute-force confirm-streak gate in update().
        self._consecutive_good_blob_frames: int = 0

        # Set (on every camera of a controller) when that controller's fused
        # update() came up empty; cleared only once a solution is next accepted.
        # prev_pose/pose_history/vel_ema are deliberately NOT cleared on a lost
        # frame (a controller recovered by brute-force still warm-starts next
        # frame from whatever history survives), so prev_pose stays non-None
        # straight through a loss — this flag is what tells finalize_search() to
        # use the loose re-acquisition check instead of the tight per-frame
        # pose-jump guard for the frame that actually recovers.
        self.tracking_lost_last_frame: bool = False

        self._matching_cfg = matching_cfg or {}

        _window = int(self._matching_cfg.get('pose_history_window', 5))
        self.pose_history: deque = deque(maxlen=_window)
        self.vel_ema: Optional[np.ndarray] = None  # shape (3,) float32, m/frame

        # Lazy cache (e.g. KD-tree later)
        self.kd_tree_cache = None

        from src.pose_search import PoseSearcher
        self._pose_searcher = PoseSearcher(camera, model, geometry_cfg, matching_cfg)
        self._geometry = self._pose_searcher._geometry

        self.proximity_match         = self._pose_searcher.proximity_search
        self.brute_match             = self._pose_searcher.brute_search
        self.prior_constrained_match = self._pose_searcher.constrained_search

    def clear_prior(self) -> None:
        """Drop everything this tracker would otherwise warm-start the next
        search from -- shared by ControllerTracker._mark_all_lost's grace-
        exhausted branch and _commit_fused_solution's persistent-reject
        escape hatch (plan Phase 5), so the two "this tracker is genuinely
        cold now" paths can't silently diverge (found in review)."""
        self.prev_pose      = None
        self.prev_prev_pose = None
        self.vel_ema        = None
        self.pose_history.clear()

    # # -----------------------------------------------------
    # # Custom projection
    # # -----------------------------------------------------
    # def project_leds_to_image(
    #         self,
    #         T_cam_ctrl: Transform  # controller → camera
    # ) -> List[Tuple[int, Optional[np.ndarray], bool]]:
    #
    #     projected = []
    #
    #     for idx, (pos, normal) in enumerate(zip(self.model.positions, self.model.normals)):
    #
    #         # --- transform to camera ---
    #         led_cam = T_cam_ctrl.apply(pos[None])[0]
    #         z = float(led_cam[2])
    #
    #         if z <= 1e-6:
    #             projected.append((idx, None, False))
    #             continue
    #
    #         # --- normalized coordinates ---
    #         x = led_cam[0] / z
    #         y = led_cam[1] / z
    #
    #         # --- distortion ---
    #         r2 = x * x + y * y
    #         r4 = r2 * r2
    #         r6 = r2 * r4
    #
    #         cam = self.camera
    #
    #         radial = (
    #                 (1 + cam.k1 * r2 + cam.k2 * r4 + cam.k3 * r6) /
    #                 (1 + cam.k4 * r2 + cam.k5 * r4 + cam.k6 * r6)
    #         )
    #
    #         dx = 2 * cam.p1 * x * y + cam.p2 * (r2 + 2 * x * x)
    #         dy = cam.p1 * (r2 + 2 * y * y) + 2 * cam.p2 * x * y
    #
    #         x_dist = x * radial + dx
    #         y_dist = y * radial + dy
    #
    #         # --- pixel ---
    #         u = cam.fx * x_dist + cam.cx
    #         v = cam.fy * y_dist + cam.cy
    #
    #         # --- visibility ---
    #         normal_cam = T_cam_ctrl.R @ normal
    #         view_dir = -led_cam / np.linalg.norm(led_cam)
    #
    #         is_visible = normal_cam @ view_dir > 0.2
    #
    #         projected.append((idx, np.array([u, v], dtype=np.float32), is_visible))
    #
    #     return projected

    # -----------------------------------------------------
    # Pose-jump guard
    # -----------------------------------------------------
    @staticmethod
    def _pose_jump_too_large(
        rvec_new, tvec_new,
        rvec_ref, tvec_ref,
        max_dist_m: float = 0.15,
        max_angle_deg: float = 25.0,
        pos_thresh_xyz_m: tuple = None,
        rot_thresh_xyz_deg: tuple = None,
    ) -> bool:
        """
        Return True if the new pose is implausibly far from the reference.

        Scalar mode (default): Euclidean translation distance and total rotation angle.
        Per-axis mode: if pos_thresh_xyz_m or rot_thresh_xyz_deg are given, each
          axis is checked independently (any axis over threshold → reject).
          For rotation the axis errors come from the Rodrigues log of the relative
          rotation, i.e. the rotation-vector components in radians.
          Per-axis mode overrides the corresponding scalar check when provided.
        """
        tvec_new = np.asarray(tvec_new, dtype=np.float64).reshape(3)
        tvec_ref = np.asarray(tvec_ref, dtype=np.float64).reshape(3)
        pos_diff = tvec_new - tvec_ref

        if pos_thresh_xyz_m is not None:
            tx, ty, tz = pos_thresh_xyz_m
            if abs(pos_diff[0]) > tx or abs(pos_diff[1]) > ty or abs(pos_diff[2]) > tz:
                return True
        elif np.linalg.norm(pos_diff) > max_dist_m:
            return True

        R_new, _ = cv2.Rodrigues(np.asarray(rvec_new, dtype=np.float32).reshape(3, 1))
        R_ref, _ = cv2.Rodrigues(np.asarray(rvec_ref, dtype=np.float32).reshape(3, 1))

        if rot_thresh_xyz_deg is not None:
            # Rodrigues log of relative rotation gives axis-angle as a 3-vector (radians).
            R_rel = R_new @ R_ref.T
            rvec_rel, _ = cv2.Rodrigues(R_rel.astype(np.float32))
            rot_deg = np.degrees(np.abs(rvec_rel.reshape(3)))
            rx, ry, rz = rot_thresh_xyz_deg
            if rot_deg[0] > rx or rot_deg[1] > ry or rot_deg[2] > rz:
                return True
        else:
            cos_a = np.clip((np.trace(R_new @ R_ref.T) - 1.0) / 2.0, -1.0, 1.0)
            if float(np.degrees(np.arccos(cos_a))) > max_angle_deg:
                return True

        return False

    @staticmethod
    def _predict_pose(
        pose_history,
        target_ts_ns: int,
        weight_decay: float = 0.7,
        vel_ema_rate: Optional[np.ndarray] = None,
        gyro_rel_R: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Predict pose at `target_ts_ns` from pose history.

        pose_history[0] = most recent (rvec, tvec, ts_ns); index increases toward
        older frames. Real capture intervals are NOT uniform (consecutive frame
        gaps can alternate between substantially different durations) — every
        branch below extrapolates against actual elapsed time, never "one
        history slot ahead".

        vel_ema_rate: if provided, a position-per-second rate (EMA-smoothed)
                 that overrides translation prediction:
                 pose_history[0].tvec + vel_ema_rate * dt_target.

        gyro_rel_R: if provided, a (3,3) body-frame relative rotation integrated
                 directly from measured controller gyro over
                 [pose_history[0]'s timestamp, target_ts_ns] (see
                 src/imu_data.integrate_gyro_segment / _gyro_rel_R_for). REPLACES
                 rotation prediction entirely — R_pred = R_0 @ gyro_rel_R — for
                 every branch below (n==1 constant / n==2 fractional / n>=3
                 linear-fit rotation logic is skipped whenever this is given).
                 It uses measured angular velocity through the real gap instead
                 of extrapolating 2-3 old vision poses, which assumes roughly
                 constant angular velocity — exactly the assumption fast hand
                 motion breaks. Translation prediction is untouched either way:
                 accelerometer-based position prediction needs gravity/bias
                 separated from real motion, which isn't safe to do without a
                 real filter (see Stage 4 plan) — Stage 3 only takes the
                 gyro/rotation half of the win.

        n=0 → None (no information)
        n=1 → constant position (same pose) — vel_ema_rate is provably always
              None here in practice (nothing has been tracked long enough yet
              to populate it), kept for defensive symmetry only.
        n=2 → constant-velocity extrapolation scaled by the ratio of the target
              gap to the one known historical gap (matrix-based rotation,
              scaled via a fractional Rodrigues vector — see below).
        n>=3 → weighted degree-1 (linear) fit over real elapsed time; exponential
               weights (weight_decay^i, still per history slot, not per elapsed
               time — see module notes on why that's a deliberate simplification).
               Computes a weighted mean velocity across the window — averaging out
               the alternating big/small step oscillation from fast hand motion.

        Degenerate timestamps (dt_hist <= 0 from duplicate/out-of-order frames,
        or dt_target <= 0 from deep-debug mode's non-consecutive replay) fall
        back to a one-unit-step assumption rather than dividing by zero or
        extrapolating backwards.
        """
        n = len(pose_history)

        if n == 0:
            return None

        R_0, _ = cv2.Rodrigues(np.asarray(pose_history[0][0], np.float32).reshape(3, 1))

        if n == 1:
            tvec_pred = np.asarray(pose_history[0][1], np.float32).reshape(3)
            if gyro_rel_R is not None:
                rvec_pred, _ = cv2.Rodrigues((R_0 @ gyro_rel_R).astype(np.float32))
                return rvec_pred.reshape(3, 1).astype(np.float32), tvec_pred
            return (
                np.asarray(pose_history[0][0], np.float32).reshape(3, 1),
                tvec_pred,
            )

        ts0 = int(pose_history[0][2])
        dt_target = (target_ts_ns - ts0) / 1e9  # seconds; may be <=0, guarded per branch

        if vel_ema_rate is not None:
            tvec_pred = (np.asarray(pose_history[0][1], np.float32).reshape(3)
                         + vel_ema_rate.reshape(3) * dt_target).astype(np.float32)
        else:
            tvec_pred = None  # filled in by the branch below

        if n == 2:
            tvec_n   = np.asarray(pose_history[0][1], np.float64).reshape(3)
            rvec_nm1 = np.asarray(pose_history[1][0], np.float32).reshape(3, 1)
            tvec_nm1 = np.asarray(pose_history[1][1], np.float64).reshape(3)
            ts_nm1   = int(pose_history[1][2])

            dt_hist = (ts0 - ts_nm1) / 1e9
            # frac=1.0 reproduces the old "assume one uniform step" behavior
            # exactly, for degenerate timestamps only.
            frac = (dt_target / dt_hist) if (dt_hist > 0 and dt_target > 0) else 1.0

            if tvec_pred is None:
                tvec_pred = (tvec_n + (tvec_n - tvec_nm1) * frac).astype(np.float32)

            if gyro_rel_R is not None:
                rvec_pred, _ = cv2.Rodrigues((R_0 @ gyro_rel_R).astype(np.float32))
                return rvec_pred.reshape(3, 1).astype(np.float32), tvec_pred.reshape(3)

            R_nm1, _ = cv2.Rodrigues(rvec_nm1)
            # Fractional rotation: scale the relative-rotation Rodrigues vector by
            # frac (small-angle-safe "fraction of a rotation") instead of always
            # applying the full historical step once more regardless of gap size.
            rvec_rel, _ = cv2.Rodrigues((R_0 @ R_nm1.T).astype(np.float32))
            rvec_rel_scaled = (rvec_rel.reshape(3) * frac).astype(np.float32)
            R_rel_scaled, _ = cv2.Rodrigues(rvec_rel_scaled.reshape(3, 1))
            R_pred = R_rel_scaled @ R_0
            rvec_pred, _ = cv2.Rodrigues(R_pred.astype(np.float32))

            return rvec_pred.reshape(3, 1).astype(np.float32), tvec_pred.reshape(3)

        # n >= 3: weighted degree-1 (linear) fit — estimates a single average velocity
        # across the window. Degree-2 would add an acceleration term that amplifies
        # the alternating big/small step oscillation typical of fast hand motion.
        # time axis: real elapsed seconds from the most recent pose (t=0, negative
        # for older frames); predict at t=dt_target — the actual elapsed time to the
        # frame being predicted for, not always +1.
        t_pts   = np.array([(int(p[2]) - ts0) / 1e9 for p in pose_history], dtype=np.float64)
        weights = weight_decay ** np.arange(n, dtype=np.float64)
        _t_eval = dt_target if dt_target > 0 else 1.0

        # Translation: linear fit per axis (skipped when vel_ema already set tvec_pred)
        if tvec_pred is None:
            tvecs = np.stack([np.asarray(p[1], np.float64).reshape(3) for p in pose_history])
            tvec_pred = np.empty(3, dtype=np.float32)
            for ax in range(3):
                tvec_pred[ax] = np.polyval(np.polyfit(t_pts, tvecs[:, ax], deg=1, w=weights), _t_eval)

        if gyro_rel_R is not None:
            rvec_pred, _ = cv2.Rodrigues((R_0 @ gyro_rel_R).astype(np.float32))
            return rvec_pred.reshape(3, 1).astype(np.float32), tvec_pred.reshape(3)

        # Rotation: linear fit in the tangent space of R_0 (most recent rotation).
        # rel_rvecs[i] = log(R_0^T @ R_i) — rotation from current pose back to the i-th
        # historical pose, expressed as a Rodrigues vector. These are always small-angle
        # deltas and avoid the ±π discontinuity of fitting absolute Rodrigues components.
        rel_rvecs = np.zeros((n, 3), dtype=np.float64)  # rel_rvecs[0] = [0,0,0] by definition
        for i in range(1, n):
            R_i, _ = cv2.Rodrigues(np.asarray(pose_history[i][0], np.float32).reshape(3, 1))
            rv, _ = cv2.Rodrigues((R_0.T @ R_i).astype(np.float32))
            rel_rvecs[i] = rv.reshape(3)

        rvec_rel_pred = np.empty(3, dtype=np.float32)
        for ax in range(3):
            rvec_rel_pred[ax] = np.polyval(np.polyfit(t_pts, rel_rvecs[:, ax], deg=1, w=weights), _t_eval)

        R_rel_pred, _ = cv2.Rodrigues(rvec_rel_pred.reshape(3, 1).astype(np.float32))
        rvec_pred, _ = cv2.Rodrigues((R_0 @ R_rel_pred).astype(np.float32))

        return rvec_pred.reshape(3, 1).astype(np.float32), tvec_pred.reshape(3)

    # -----------------------------------------------------
    # Tracking
    # -----------------------------------------------------
    def search_cheap(self, blobs: np.ndarray, frame_ts_ns: int,
                      blob_radii: Optional[np.ndarray] = None,
                      blob_brightnesses: Optional[np.ndarray] = None,
                      other_cameras_blobs: Optional[List] = None,
                      blob_mask: Optional[np.ndarray] = None,
                      occluders_per_cam: Optional[Dict] = None,
                      gyro_rel_R: Optional[np.ndarray] = None,
                      vel_ema_override: Optional[np.ndarray] = None,
                      ) -> Tuple[Optional[Dict], Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Proximity + prior_constrained only — no brute-force. Reads self state (does
        not commit). Returns (solution_or_None, predicted_pose).

        A None solution here means this camera needs brute-force recovery to get any
        candidate at all this frame — whether because it has no prior (cold-start) or
        because proximity/prior_constrained came up empty despite having one. Callers
        doing cross-camera recovery should treat both cases identically: proceed to
        brute-force with pose_prior=predicted_pose (None for cold-start, matching
        today's unprimed cold-start brute call).

        gyro_rel_R: optional (3,3) body-frame relative rotation from measured
        controller gyro (see _gyro_rel_R_for) — caller computes this (it needs
        ctrl_name to pick the right IMU stream, which this per-camera object
        doesn't know) and passes it straight through to _predict_pose.

        vel_ema_override: only consulted when self.vel_ema is itself None (a
        real vision-derived vel_ema always wins) — see
        _vel_ema_with_imu_fallback's own docstring for what the caller
        computes here and why (the fusion filter's IMU dead-reckoning
        filling the one gap right after a fresh reacquisition where
        vision-only velocity isn't trustworthy yet).
        """
        prior = {
            'prev_pose':       self.prev_pose,
            'prev_prev_pose':  self.prev_prev_pose,
            'pose_history':    self.pose_history,
            'vel_ema':         self.vel_ema if self.vel_ema is not None else vel_ema_override,
            'prev_assignment': self.prev_assignment,
            'gyro_rel_R':      gyro_rel_R,
        }
        solution, predicted_pose, norm_prev, norm_prev_prev = cheap_search_core(
            self._pose_searcher, prior, self._matching_cfg,
            blobs, frame_ts_ns, blob_radii, blob_brightnesses,
            other_cameras_blobs, blob_mask, occluders_per_cam,
        )
        self.prev_pose      = norm_prev
        self.prev_prev_pose = norm_prev_prev
        return solution, predicted_pose

    def finalize_search(self, solution: Optional[Dict],
                         predicted_pose: Optional[Tuple[np.ndarray, np.ndarray]],
                         blobs: np.ndarray, blob_radii: Optional[np.ndarray] = None,
                         other_cameras_blobs: Optional[List] = None,
                         blob_mask: Optional[np.ndarray] = None,
                         occluders_per_cam: Optional[Dict] = None,
                         allow_expensive_fallback: bool = True,
                         *, frame_ts_ns: int) -> Optional[Dict]:
        """Validate and accept/reject an already-obtained candidate `solution` — from
        search_cheap(), or from a brute-force recovery attempt (cold-start or
        proximity-failed). Reads self state, does not commit results — call apply()
        with the returned value to commit state changes.

        frame_ts_ns: this frame's real timestamp -- keyword-only (forces every call
        site to pass it explicitly rather than silently misaligning positionally).
        Needed by the cold-start staleness widening below, which must use REAL
        elapsed time since last_good_pose was captured, not a frame-count proxy
        (this recording's own frame rate oscillates ~11-22ms, so a count-based
        proxy diverges from the true elapsed gap -- found auditing every
        prediction/staleness site in this pipeline for exactly this class of bug).

        allow_expensive_fallback: gates the brute-force retries that fire when this
        camera HAS a prior but the candidate is a pose-jump or its error is too high —
        brute-force can cost hundreds of ms for no result. Callers with other cameras
        available this frame should pass False on a first pass and only retry with
        True if the whole controller came up empty — a struggling camera can otherwise
        be warm-started next frame from a fused pose at zero cost.
        """
        blobs   = np.asarray(blobs, dtype=np.float32).reshape(-1, 2)
        n_blobs = len(blobs)
        n_available = int(blob_mask.sum()) if blob_mask is not None else n_blobs

        cam_idx = self.camera.camera_idx
        ctrl_name = self.model.name.replace("_controller", "")

        _cfg = self._matching_cfg
        _accept_err_px = float(_cfg.get('accept_error_px', 3.0))
        _pos_ax = _cfg.get('pose_jump_pos_thresh_m')
        _rot_ax = _cfg.get('pose_jump_rot_thresh_deg')
        _jump_kw = {}
        if _pos_ax is not None:
            _jump_kw['pos_thresh_xyz_m']   = tuple(_pos_ax)
        if _rot_ax is not None:
            _jump_kw['rot_thresh_xyz_deg'] = tuple(_rot_ax)

        # Re-acquisition: either a genuine cold start (prev_pose is None — not
        # reachable in practice today, kept for completeness) or the first
        # solution accepted after tracking_lost_last_frame was set. prev_pose/
        # pose_history are deliberately left untouched by a lost frame (so a
        # recovered controller can still warm-start from whatever history
        # survives), so prev_pose being non-None is NOT by itself evidence of
        # continuous per-frame tracking — tracking_lost_last_frame is.
        _reacquiring = self.prev_pose is None or self.tracking_lost_last_frame

        # Stricter floor for a WARM/proximity accept specifically while the
        # velocity estimate is IMMATURE -- len(pose_history) < 3, i.e. fewer
        # than pose_history_window's own "3+=linear fit" tier (see its config
        # comment: 1=constant pos, 2=const vel, 3+=linear fit). Cheap/
        # proximity search builds its per-LED candidate neighborhoods from
        # predicted_pose, which at this depth is either a constant-position +
        # gyro-only extrapolation (n=1, self.vel_ema is None -- see its own
        # comment below) or a single-pair, UNSMOOTHED velocity estimate (n=2,
        # self.vel_ema is that one raw step -- no EMA blending has happened
        # yet, see apply()'s own vel_ema computation) -- with no mature
        # velocity to work from, a controller that moved at all (n=1) OR a
        # single noisy step inherited from an imprecise anchor pose (n=2) gets
        # neighborhoods centered on the WRONG spot, and proximity's own
        # combinatorial search degrades accordingly. Confirmed on a real
        # recording (frame_range 3850-3950 relative frame 51): pose_history
        # was at n=2 (vel_ema a single raw step derived from the PRIOR frame's
        # own weak 5-inlier/0.98px accept), 9 of 13 visible LEDs' neighborhoods
        # found zero candidates at all, the primary top-k search hit its own
        # 8000-node safety cap without finding a result, and the eventual
        # fallback "best" hypothesis matched only 4/13 LEDs at up to 2.07px
        # residual -- yet nothing rejected it, because the position half of
        # the jump gate is deliberately disabled/loosened at this depth (see
        # self.vel_ema is None's own comment below) and rotation alone
        # happened to agree. Rather than loosen a check that's correctly
        # relaxed for a real reason, this instead raises the bar for what
        # counts as an acceptable proximity result in this one narrow window:
        # it must clear strong_match_inliers/strong_match_error_px (the same
        # "confidently strong" bar pose_search.py's own strong_found decision
        # and the last_good_pose quality-rescue above both already use) on
        # its own merits, with no prediction-based check to lean on instead.
        # Failing that floor is treated as "no solution found" here, exactly
        # like an outright proximity failure -- main.py's existing "warm
        # proximity lost" fallback then runs a fresh COLD (brute-force) re-
        # detect, which doesn't depend on this same unreliable prediction at
        # all. Brute-force/cold results are deliberately EXEMPT: their own
        # coverage-fallback mechanism ("take the best available candidate
        # rather than nothing") is an intentional design choice for a genuine
        # cold start with no other reference to fall back on -- this floor
        # would defeat that policy for no reason, since a rejected cold
        # result has nowhere cheaper left to fall back to.
        if (solution is not None and len(self.pose_history) < 3
                and solution.get("method") == "proximity"):
            _n_inliers_floor = (len(solution.get("assignment") or [])
                                  + sum(len(v) for v in (solution.get("aux_assignments") or {}).values()))
            _strong_inliers_floor = float(_cfg.get("strong_match_inliers", 6))
            _strong_error_floor_px = float(_cfg.get("strong_match_error_px", 0.5))
            if (_n_inliers_floor < _strong_inliers_floor
                    or float(solution.get("error", float("inf"))) > _strong_error_floor_px):
                logger.bind(cat="matching_decisions").debug(
                    f"[{ctrl_name} | cam {cam_idx} | track] proximity accept rejected: below "
                    f"strong-match floor with an immature velocity estimate "
                    f"(pose_history_len={len(self.pose_history)}, {_n_inliers_floor} "
                    f"inliers/{float(solution.get('error', float('inf'))):.2f}px, need "
                    f">= {_strong_inliers_floor:.0f}/<= {_strong_error_floor_px:.2f}px)"
                )
                solution = None

        # Temporary diagnostic (see _log_jump_stats_row, no-op unless
        # CONTROLLER_TRACKER_JUMP_STATS_CSV is set) -- captures the raw
        # predicted-vs-actual distribution for continuous warm frames only
        # (the exact population the tight per-frame jump gate below applies
        # to), BEFORE either jump gate can reject/mutate this candidate.
        if _JUMP_STATS_CSV_PATH and solution is not None and not _reacquiring and self.prev_pose is not None:
            _tv_new = np.asarray(solution["tvec"], np.float64).reshape(3)
            _R_new, _ = cv2.Rodrigues(np.asarray(solution["rvec"], np.float32).reshape(3, 1))
            _rvp, _tvp = self.prev_pose
            _tv_prev = np.asarray(_tvp, np.float64).reshape(3)
            _R_prev, _ = cv2.Rodrigues(np.asarray(_rvp, np.float32).reshape(3, 1))
            _pos_diff_prev_mm = float(np.linalg.norm(_tv_new - _tv_prev)) * 1000.0
            # R_new @ R_ref.T -- MUST match _pose_jump_too_large's own convention
            # (src/controller.py:512) exactly, not the transposed order. The total
            # rotation ANGLE (this magnitude-only use) is actually invariant to
            # the order -- R_new@R_ref.T and R_ref.T@R_new are conjugate/similar
            # matrices, same trace, same angle -- so this specific line was never
            # numerically wrong; kept consistent with the fix below anyway so
            # nothing here is left as a trap for a future per-axis extraction.
            _rot_diff_prev_deg = float(np.degrees(np.linalg.norm(
                cv2.Rodrigues((_R_new @ _R_prev.T).astype(np.float32))[0])))
            _rot_diff_prev_xyz = np.degrees(np.abs(
                cv2.Rodrigues((_R_new @ _R_prev.T).astype(np.float32))[0].reshape(3)))
            _dt_prev_s = ((frame_ts_ns - int(self.pose_history[0][2])) / 1e9
                          if self.pose_history else float("nan"))
            _pos_diff_pred_mm = _rot_diff_pred_deg = float("nan")
            _rot_diff_pred_xyz = np.array([float("nan")] * 3)
            if predicted_pose is not None:
                _rvec_pred, _tv_pred = predicted_pose  # _predict_pose returns (rvec, tvec), NOT a rotation matrix
                _R_pred, _ = cv2.Rodrigues(np.asarray(_rvec_pred, np.float32).reshape(3, 1))
                _pos_diff_pred_mm = float(np.linalg.norm(_tv_new - np.asarray(_tv_pred, np.float64).reshape(3))) * 1000.0
                _rot_diff_pred_deg = float(np.degrees(np.linalg.norm(
                    cv2.Rodrigues((_R_new @ _R_pred.T).astype(np.float32))[0])))
                _rot_diff_pred_xyz = np.degrees(np.abs(
                    cv2.Rodrigues((_R_new @ _R_pred.T).astype(np.float32))[0].reshape(3)))
            _n_inliers_stat = (len(solution.get("assignment") or [])
                                + sum(len(v) for v in (solution.get("aux_assignments") or {}).values()))
            _log_jump_stats_row(
                ts_ns=frame_ts_ns, ctrl_name=ctrl_name, cam_idx=cam_idx,
                method=solution.get("method", "?"),
                n_inliers=_n_inliers_stat, error_px=float(solution.get("error", 0.0)),
                n_pose_history=len(self.pose_history), consecutive_failures=self.consecutive_failures,
                dt_prev_s=_dt_prev_s,
                pos_diff_prev_mm=_pos_diff_prev_mm, rot_diff_prev_deg=_rot_diff_prev_deg,
                pos_diff_pred_mm=_pos_diff_pred_mm, rot_diff_pred_deg=_rot_diff_pred_deg,
                rot_diff_prev_x_deg=_rot_diff_prev_xyz[0], rot_diff_prev_y_deg=_rot_diff_prev_xyz[1], rot_diff_prev_z_deg=_rot_diff_prev_xyz[2],
                rot_diff_pred_x_deg=_rot_diff_pred_xyz[0], rot_diff_pred_y_deg=_rot_diff_pred_xyz[1], rot_diff_pred_z_deg=_rot_diff_pred_xyz[2],
                speed_est_m_s=(_pos_diff_prev_mm / 1000.0 / _dt_prev_s) if _dt_prev_s and _dt_prev_s > 0 else float("nan"),
            )

        # ------------------------------------------------------------------
        # Shared vs-predicted_pose check -- computed ONCE here, at a single
        # empirically-derived tight threshold, and referenced by BOTH the
        # cold-start/re-acquisition block below and the tight per-frame
        # guard further down. CONSOLIDATED 2026-09-10: previously each of
        # those two blocks ran its OWN separate vs-predicted_pose check with
        # a DIFFERENT threshold -- the cold-start block's rescue check
        # reused the old, much looser flat vs-prev_pose thresholds
        # ([0.18,0.18,0.20]m / [30,30,30]deg), while the tight per-frame
        # guard used the thresholds derived below -- redundant AND
        # inconsistent (predicted_pose effectively got graded on two
        # different curves depending on which block happened to run).
        # predicted_pose's ROLE (independent veto vs. OR-rescue) still
        # differs by regime -- see the tight per-frame guard's own comment
        # further down for why -- only the underlying check itself, and its
        # threshold, are now unified into one.
        #
        # vs-predicted_pose position AND rotation tolerance are both
        # VELOCITY-DEPENDENT (not the fixed pos_thresh_xyz_m /
        # rot_thresh_xyz_deg used for the vs-prev_pose check below) --
        # predicted_pose is a genuinely better estimator than prev_pose (it
        # extrapolates velocity and, for rotation, integrates real measured
        # gyro -- see _predict_pose's own docstring), so its residual
        # against the eventual real solve is both smaller AND more
        # speed-dependent than prev_pose's raw frame-to-frame delta, and
        # deserves its own tighter, separately-derived bound rather than
        # reusing vs-prev_pose's threshold.
        #
        # Position re-derived 2026-09-10 (rotation ADDED 2026-09-10, never
        # had its own threshold before -- see git history) against the FULL
        # walk_medium recording this time (7335 frames, both controllers,
        # CONTROLLER_TRACKER_JUMP_STATS_CSV, src/controller.py), not just a
        # sub-range: 7295 quality-filtered samples (n_inliers>=8,
        # error_px<=0.8, same bar as the original 2026-09-06 tuning).
        # Per-speed-bin MAX (not a percentile -- percentiles under-cover by
        # construction) of the real predicted-vs-actual residual, linearly
        # fit against bin-mean speed, then margined until the formula clears
        # 100% of the 7295 quality samples with zero false-rejects
        # (re-verified directly, row by row, not estimated):
        #   position   (magnitude): raw max-fit ~= 30.7 + 16.5*speed_m/s mm
        #                            -- needed a full 2x margin for zero
        #                            false-rejects on this larger, more
        #                            representative sample. The OLD 10+10
        #                            default (fit against a smaller
        #                            sub-range sample) turned out to already
        #                            be TOO TIGHT at full-recording scale: it
        #                            was silently false-rejecting 54/7295
        #                            (0.74%) of genuinely good frames -- a
        #                            real, live miscalibration.
        #   rotation x (per-axis): raw max-fit ~= 6.5 + 0.42*speed deg,
        #                            x1.3 margin, zero false-rejects.
        #   rotation y (per-axis): raw max-fit ~= 8.7 + 0.98*speed deg,
        #                            x1.4 margin, zero false-rejects (the
        #                            noisiest axis -- also the one that let
        #                            a real ~38deg fusion-level jump through
        #                            undetected before this fix, since
        #                            vs-predicted_pose previously had no
        #                            rotation threshold of its own at all).
        #   rotation z (per-axis): raw max-fit ~= 3.5 + 0.50*speed deg,
        #                            x1.3 margin, zero false-rejects.
        # Checked and explicitly ruled out as confounds before trusting this
        # fit: (a) n_inliers within the quality set barely affects the
        # residual tail (max at n_inliers=8 vs 9-13 is ~13.3 vs ~13.2deg --
        # the >=8 gate itself already does the real work); (b) pose_history
        # maturity (pose_history_window=3 -- see its own config comment)
        # does NOT explain the outlier tail either: 99% of quality samples
        # are already at full window depth, and the tiny immature (<3)
        # subset actually has a LOWER max residual (46mm) than the mature
        # one (97mm) -- ruled out as the general explanation (though it was
        # real for the one ~38deg case that motivated re-deriving this in
        # the first place). self.vel_ema is the SAME velocity estimate
        # predicted_pose itself was built from this frame (see
        # _predict_pose's vel_ema_rate param) -- not a separate estimate.
        # ------------------------------------------------------------------
        _speed_est_m_s = float(np.linalg.norm(self.vel_ema)) if self.vel_ema is not None else 0.0
        _vel_pos_thresh_m = (float(_cfg.get('pose_jump_pred_pos_thresh_base_mm', 61.5))
                              + float(_cfg.get('pose_jump_pred_pos_thresh_per_speed_mm_s', 33.0))
                              * _speed_est_m_s) / 1000.0
        _pred_rot_base_deg      = _cfg.get('pose_jump_pred_rot_thresh_base_deg', [8.4, 12.1, 4.5])
        _pred_rot_per_speed_deg = _cfg.get('pose_jump_pred_rot_thresh_per_speed_deg_s', [0.55, 1.37, 0.65])
        _vel_rot_thresh_deg = tuple(
            float(b) + float(p) * _speed_est_m_s
            for b, p in zip(_pred_rot_base_deg, _pred_rot_per_speed_deg)
        )
        # Extra margin while RE-ACQUIRING specifically (added 2026-09-10):
        # during a real loss, predicted_pose is built from the fusion
        # filter's raw IMU-only dead-reckoning (ControllerTracker.
        # _mark_all_lost), not the vision-history-based extrapolation the
        # base thresholds above were empirically fit against -- a DIFFERENT
        # error regime whose uncertainty grows with how long we've been
        # coasting on IMU alone, not just with speed. Widens by real elapsed
        # time since the last real vision accept (last_good_pose_ts_ns --
        # the same anchor _mark_all_lost's own IMU-only budget and the
        # cold-start block's own last_good_pose widening below both use),
        # reusing THOSE SAME rates (cold_start_stale_max_speed_m_s /
        # _max_ang_speed_deg_s) for consistency. NOT independently
        # re-derived/validated against real IMU-only-coast data the way the
        # base thresholds above were (that would need its own
        # CONTROLLER_TRACKER_JUMP_STATS_CSV-style empirical pass) -- a
        # reasonable, physically-motivated conservative margin, not an
        # empirically-tight one. Zero during continuous tracking
        # (_reacquiring False), where the base thresholds already apply
        # unwidened, exactly as empirically validated.
        if _reacquiring and self.last_good_pose_ts_ns is not None:
            _coast_stale_s = max(0.0, (frame_ts_ns - self.last_good_pose_ts_ns) / 1e9)
            _vel_pos_thresh_m += float(_cfg.get('cold_start_stale_max_speed_m_s', 3.0)) * _coast_stale_s
            _extra_pred_rot_deg = float(_cfg.get('cold_start_stale_max_ang_speed_deg_s', 720.0)) * _coast_stale_s
            _vel_rot_thresh_deg = tuple(v + _extra_pred_rot_deg for v in _vel_rot_thresh_deg)
        # No established velocity estimate yet (self.vel_ema is None -- true
        # exactly when pose_history has fewer than 2 accepted frames):
        # _predict_pose's translation half is NOT a prediction here, it
        # returns pose_history[0]'s own tvec completely unchanged (see its
        # own n==1 docstring branch) -- the SAME reference point vs_prev_pose
        # already checks. Left as _vel_pos_thresh_m's base-only 61.5mm, that
        # identical displacement would be silently re-judged against a
        # threshold far TIGHTER than vs_prev_pose's own (180/180/200mm), with
        # no velocity information to justify the extra strictness -- purely
        # an artifact of which check happens to run, not a real second
        # opinion. Disabled outright (not widened to some other number worth
        # re-litigating later) rather than picked to match vs_prev_pose's own
        # bound, which would just be re-deriving a threshold that already
        # exists one call away. Rotation is untouched: it genuinely differs
        # here (gyro-measured, independent of vision-history depth -- see
        # _predict_pose's own docstring) and keeps doing real work regardless
        # of vel_ema. Confirmed on a real recording (frame_range 1000-1010,
        # relative frame 4): a candidate 90mm from prev_pose (well inside
        # vs_prev_pose's own 180mm) was independently vetoed by this position
        # check alone at n=1 -- vetoed anyway that frame by rotation, but a
        # position-only false reject at n=1 was otherwise fully possible.
        if self.vel_ema is None:
            _vel_pos_thresh_m = float('inf')
        _jump_vs_pred = None
        if solution is not None and predicted_pose is not None:
            _jump_vs_pred = self._pose_jump_too_large(
                solution["rvec"], solution["tvec"],
                predicted_pose[0], predicted_pose[1],
                max_dist_m=_vel_pos_thresh_m,
                rot_thresh_xyz_deg=_vel_rot_thresh_deg,
            )

        # Cold-start / re-acquisition plausibility check: reject a candidate
        # that's too far from BOTH the last known good pose AND the current
        # predicted (IMU-extrapolated) pose. Only valid when frames are a real
        # temporal sequence — against a curated/isolated frame set
        # (assume_continuous_frames: false) there's no real "last frame" to be
        # near, so this check is skipped and the controller can be anywhere.
        #
        # No hard frame-count cutoff any more (previously
        # tracking_lost_grace_frames=1, then imu_only_propagation_max_frames=4
        # -- see git history for both). Both were hard cliffs: tight check
        # right up to the cutoff, then NO CHECK AT ALL past it. Found a real
        # case where that gap mattered even at 4 frames: a 6-consecutive-lost-
        # frame gap (one past that budget) let a confidently-wrong, thin
        # brute-force bootstrap (4 inliers, 0.05 confidence, 14/32 LED triples
        # reached) sail through completely unchecked. Replaced with a
        # continuous widening instead:
        #
        #   - Up to imu_only_propagation_max_s of real elapsed time since the
        #     fusion filter's last accepted vision update: predicted_pose
        #     (IMU-extrapolated, computed fresh THIS frame regardless of how
        #     long the loss has run) is still live -- see
        #     ControllerTracker._mark_all_lost, which keeps THIS camera's
        #     prev_pose/pose_history IMU-warm-started for exactly that long.
        #     Checked at the normal tight threshold (never stale by
        #     construction -- it's a fresh prediction every call).
        #   - Past that budget, pose_history gets cleared (clear_prior()) and
        #     predicted_pose goes None in lockstep -- last_good_pose (frozen at
        #     the last real vision accept) is all that's left, and it gets
        #     staler every additional REAL SECOND, not every additional lost
        #     FRAME -- this recording's own frame rate oscillates (~11-22ms
        #     gaps), so a frame-count proxy for elapsed time diverges from the
        #     truth over a multi-frame loss (an earlier version of this fix
        #     used consecutive_failures * an assumed nominal frame period;
        #     found wrong auditing every prediction/staleness site in this
        #     pipeline for exactly this class of bug -- fixed by tracking
        #     last_good_pose_ts_ns alongside last_good_pose, see its own
        #     comment). Threshold WIDENS continuously with REAL elapsed time
        #     since last_good_pose (cold_start_stale_max_speed_m_s /
        #     _max_ang_speed_deg_s, a generous fast-hand-motion-scale rate) --
        #     NOT gated on first crossing the imu_only_propagation_max_s
        #     budget (a second, later bug in the first version of this fix,
        #     back when the IMU-only budget was still frame-counted: that gate
        #     assumed real elapsed time and the budget's frame count move
        #     together, which breaks during a fast-cadence stretch -- found on
        #     a real case where consecutive_failures=8, already past the
        #     4-frame budget so predicted_pose was gone, but real elapsed time
        #     was still under the budget's assumed duration, so a clean
        #     6-inlier/0.04px candidate got "+0mm/+0deg for 0.00s stale" --
        #     the exact same failure this whole mechanism exists to prevent.
        #     The IMU-only budget itself was converted to real time
        #     (imu_only_propagation_max_s, 2026-09-09) specifically so it and
        #     this widening's own _stale_s always read off the same clock --
        #     see that config key's own comment). A continuous version of the
        #     ORIGINAL comment's own reasoning ("may have had arbitrarily long
        #     -- and thus arbitrarily far -- to drift, so the check no longer
        #     applies"), arrived at gradually instead of via a sudden binary
        #     cliff, so an extremely long loss still degrades toward "no
        #     meaningful check" the same way it always did, just without a
        #     specific point where protection vanishes outright.
        #
        # Same OR-logic as the tight per-frame guard below (accept if
        # plausible against EITHER reference) -- kept consistent with that
        # check deliberately, not reinvented here.
        _cold_start_check_budget_s = float(_cfg.get('imu_only_propagation_max_s', 0.066))
        _check_vs_last_good = (_reacquiring and self.last_good_pose is not None
                                and is_continuous_sequence())
        if solution is not None and _check_vs_last_good:
            rvec_lg, tvec_lg = self.last_good_pose

            # Real elapsed time since last_good_pose was captured -- widens
            # from t=0, NO baseline "free pass" window. An earlier version of
            # this fix subtracted imu_only_propagation_max_frames worth of
            # assumed time before widening kicked in (reasoning: predicted_pose
            # covers that regime via the OR-check below anyway) -- wrong,
            # found on a real case: consecutive_failures=8 (well past the
            # 4-frame budget, so predicted_pose was ALREADY unavailable) but
            # real elapsed time still under that ~133ms baseline (a fast-
            # cadence stretch), so a clean 6-inlier/0.04px candidate got
            # rejected against last_good_pose with "+0mm/+0deg for 0.00s
            # stale" -- the tight, ~1-frame-sized base threshold, even though
            # up to 8 real (undefended) frames had actually elapsed and
            # predicted_pose was no longer there to rescue it either. The
            # baseline conflated two different budgets (predicted_pose
            # availability vs. how much widening the FIXED reference pose
            # needs) that don't actually move together once real cadence
            # varies -- removed; widening now tracks the true elapsed time
            # unconditionally, so it can never lag behind predicted_pose's own
            # (frame-count-based) expiry the way the baseline let it.
            _stale_s = 0.0
            if self.last_good_pose_ts_ns is not None:
                _stale_s = max(0.0, (frame_ts_ns - self.last_good_pose_ts_ns) / 1e9)
            _extra_pos_m = float(_cfg.get('cold_start_stale_max_speed_m_s', 3.0)) * _stale_s
            _extra_rot_deg = float(_cfg.get('cold_start_stale_max_ang_speed_deg_s', 720.0)) * _stale_s

            def _widen(_thresh, _extra):
                return (tuple(v + _extra for v in _thresh) if isinstance(_thresh, tuple)
                        else _thresh + _extra)

            _lg_jump_kw = dict(_jump_kw)
            if 'pos_thresh_xyz_m' in _lg_jump_kw:
                _lg_jump_kw['pos_thresh_xyz_m'] = _widen(_lg_jump_kw['pos_thresh_xyz_m'], _extra_pos_m)
            if 'rot_thresh_xyz_deg' in _lg_jump_kw:
                _lg_jump_kw['rot_thresh_xyz_deg'] = _widen(_lg_jump_kw['rot_thresh_xyz_deg'], _extra_rot_deg)

            _jump_vs_last_good = self._pose_jump_too_large(
                solution["rvec"], solution["tvec"],
                rvec_lg, tvec_lg,
                max_dist_m=0.5 + _extra_pos_m,
                max_angle_deg=60.0 + _extra_rot_deg,
                **_lg_jump_kw,
            )

            # Quality-comparison rescue: a candidate that CLEARLY beats
            # last_good_pose's own recorded quality on both inliers and error
            # is evidence the REFERENCE was the inaccurate one, not that the
            # candidate is an implausible jump -- see last_good_pose_quality's
            # own comment (__init__) for the real case this fixes (a clean
            # 11-inlier/0.12px candidate rejected against a 5-inlier/0.98px
            # COVERAGE-FALLBACK reference by 2.6deg on one rotation axis).
            # Deliberately conservative on both axes at once (not "OR"): a
            # candidate that's merely somewhat better shouldn't override this
            # gate on a hunch, only one so clearly stronger that the
            # reference's own inaccuracy is the more likely explanation.
            # "Strong" reuses strong_match_inliers/strong_match_error_px
            # (pose_search.py's own bar for a confident match) rather than a
            # new tunable -- the candidate must clear that bar outright, not
            # just beat the reference by some relative margin, so a weak
            # reference can never validate an only-slightly-less-weak
            # candidate as "clearly better."
            _quality_rescue = False
            if _jump_vs_last_good and self.last_good_pose_quality is not None:
                _lg_inliers, _lg_error = self.last_good_pose_quality
                _cand_inliers = len(solution.get("assignment") or [])
                _cand_error = float(solution.get("error", float("inf")))
                _strong_inliers = float(_cfg.get("strong_match_inliers", 6))
                _strong_error_px = float(_cfg.get("strong_match_error_px", 0.5))
                _reference_weak = _lg_inliers < _strong_inliers or _lg_error > _strong_error_px
                _candidate_strong = _cand_inliers >= _strong_inliers and _cand_error <= _strong_error_px
                _candidate_clearly_better = (_cand_inliers >= 2 * _lg_inliers
                                              and _cand_error <= 0.5 * max(_lg_error, 1e-6))
                _quality_rescue = _reference_weak and _candidate_strong and _candidate_clearly_better
                if _quality_rescue:
                    logger.bind(cat="matching_decisions").debug(
                        f"[{ctrl_name} | cam {cam_idx} | track] re-acquisition candidate far from "
                        f"last_good_pose but rescued by quality comparison (candidate: "
                        f"{_cand_inliers} inliers/{_cand_error:.2f}px vs weak reference: "
                        f"{_lg_inliers} inliers/{_lg_error:.2f}px) — accepting."
                    )

            # _jump_vs_pred is the SHARED, tight vs-predicted_pose check
            # computed once above (2026-09-10 consolidation) -- previously
            # this block ran its OWN separate, much looser rescue check here.
            if _jump_vs_last_good and _quality_rescue:
                pass  # accepted above -- solution stays as-is
            elif _jump_vs_last_good and (predicted_pose is None or _jump_vs_pred):
                # stale_s vs imu_budget_s are now directly comparable (same
                # clock, same units -- see imu_only_propagation_max_s's own
                # config comment): stale_s > imu_budget_s means predicted_pose
                # is None because the IMU-only budget is simply exhausted, not
                # because a real IMU-vs-candidate distance was computed and
                # found too large -- disambiguated explicitly below rather
                # than folding both into one "and predicted (IMU) pose"
                # phrase (found confusing a real investigation: a
                # consecutive_failures=9 reject read as if an actual
                # predicted-pose distance had been checked and failed, when
                # no such comparison had run at all).
                _pred_reason = (f"no predicted pose available (past imu_budget_s={_cold_start_check_budget_s:.3f})"
                                 if predicted_pose is None else "predicted (IMU) pose also too far")
                logger.bind(cat="matching_decisions").debug(
                    f"[{ctrl_name} | cam {cam_idx} | track] brute re-acquisition rejected: too far "
                    f"from last known good pose (widened +{_extra_pos_m * 1000:.0f}mm/"
                    f"+{_extra_rot_deg:.0f}deg for stale_s={_stale_s:.3f}) and {_pred_reason} "
                    f"(consecutive_failures={self.consecutive_failures})"
                )
                solution = None
            elif _jump_vs_last_good:
                logger.bind(cat="matching_decisions").debug(
                    f"[{ctrl_name} | cam {cam_idx} | track] re-acquisition candidate far from "
                    f"last_good_pose but rescued by predicted_pose "
                    f"(consecutive_failures={self.consecutive_failures}, "
                    f"stale_s={_stale_s:.3f}, imu_budget_s={_cold_start_check_budget_s:.3f}) — accepting."
                )

        # ------------------------------------------------------------------
        # Pose-jump guard against prev_pose AND predicted_pose (tight,
        # per-frame). See the next comment block below for which arm applies
        # during re-acquisition. Rejects only if the candidate is implausibly far from
        # BOTH references, not just the raw previous frame -- prev_pose alone
        # has no notion of how fast the controller was actually moving, so a
        # genuine fast-motion frame can legitimately clear pos/rot thresholds
        # sized for "one ordinary frame step" versus prev_pose while still
        # landing close to predicted_pose (the vel_ema-extrapolated
        # prediction, which DOES account for that motion). Confirmed on a
        # real case (2026-09-06 investigation): a rotation-only jump of
        # 31.3deg on one axis vs prev_pose (threshold 30.0deg -- ~4% over,
        # position was nowhere near its own threshold) that was only 21.5deg
        # vs predicted_pose, comfortably under threshold there -- rejecting
        # it forced a brute-force recovery that failed and cost the whole
        # frame (TRACKING LOST). Same OR-logic the brute-recovery fallback a
        # few lines below already uses for its own rescue check
        # (_near_prev or _near_pred) -- this makes the INITIAL gate consistent
        # with that established pattern instead of only applying it after a
        # reject has already happened.
        # ------------------------------------------------------------------
        # vs-prev_pose only makes sense for continuous tracking (see comment
        # above _reacquiring's own definition) -- but vs-predicted_pose is
        # already velocity/time-aware (see _vel_pos_thresh_m below), so a
        # recent tracking loss doesn't make IT meaningless. Previously
        # _reacquiring gated this WHOLE block, silently skipping even the
        # vs-predicted_pose arm on the first frame back after a loss and
        # deferring straight to the much looser fusion-level
        # implausible_jump_rot_deg gate (60deg flat, not time-scaled)
        # instead. Found 2026-09-10: a reacquisition frame landed with
        # rot_innov=38.1deg vs the IMU state at dt=11.1ms (~3400deg/s -- not
        # remotely physical) and sailed through with no per-camera check at
        # all because self.prev_pose alone gated this entire block off.
        _check_vs_prev = self.prev_pose is not None and not _reacquiring
        if solution is not None and (predicted_pose is not None or _check_vs_prev):
            rvec_p = tvec_p = None
            _jump_vs_prev = False
            if _check_vs_prev:
                rvec_p, tvec_p = self.prev_pose
                _jump_vs_prev = self._pose_jump_too_large(
                    solution["rvec"], solution["tvec"],
                    rvec_p, tvec_p,
                    **_jump_kw,
                )
            # _jump_vs_pred, _vel_pos_thresh_m, _vel_rot_thresh_deg, and
            # _speed_est_m_s are the SHARED vs-predicted_pose check computed
            # once above (2026-09-10 consolidation -- see that block's own
            # comment for the full empirical derivation).
            if _check_vs_prev:
                # Continuous tracking: predicted_pose is AUTHORITATIVE when
                # available (empirically validated for exactly this regime,
                # zero false-rejects on 7295 real quality samples) -- its own
                # check alone decides, both rescuing a vs-prev_pose failure
                # (the original 2026-09-06 motivation -- prev_pose alone has
                # no notion of real motion speed) AND independently rejecting
                # a candidate that agrees with vs-prev_pose but disagrees
                # with vs-predicted_pose (found 2026-09-10: a real ~38deg
                # fusion-level jump that both cameras individually PASSED
                # vs-prev_pose on went undetected before this fix).
                # vs-prev_pose is only the fallback authority when
                # predicted_pose isn't available at all.
                _is_jump = bool(_jump_vs_pred) if predicted_pose is not None else _jump_vs_prev
            elif _check_vs_last_good:
                # Re-acquiring with a usable last_good_pose: the OR-rescue
                # against last_good_pose above (widened by elapsed staleness)
                # is the FULL decision for this regime -- predicted_pose
                # during an active loss is built from raw IMU-only dead-
                # reckoning, a different and less-validated error regime
                # (see the extra drift-proportional margin added above), so
                # it must not independently re-veto a candidate the
                # cold-start block already accepted. BUG fixed 2026-09-10:
                # this branch used to fall through to the predicted_pose-
                # alone case below, silently overriding an already-made
                # last_good_pose-based accept -- found investigating a user
                # question about accumulated IMU drift across a multi-frame
                # loss, which is exactly the scenario this bug bit hardest.
                _is_jump = False
            elif predicted_pose is not None:
                # True cold start: no prev_pose, no usable last_good_pose --
                # predicted_pose is the only reference there is, so it alone
                # decides.
                _is_jump = bool(_jump_vs_pred)
            else:
                # No reference available at all -- this check has nothing to say.
                _is_jump = False

            if _check_vs_prev and _jump_vs_prev and not _is_jump:
                # Rescued: implausible vs the raw previous frame, but
                # plausible vs the motion-compensated prediction -- accepted
                # as genuine fast motion, not a mismatch. Debug, not warning:
                # this is the expected/working case the OR-logic exists for.
                logger.bind(cat="matching_decisions").debug(
                    f"[{ctrl_name} | cam {cam_idx} | tracking] pose jump vs prev_pose rescued by "
                    f"predicted_pose (thresh={_vel_pos_thresh_m * 1000:.0f}mm @ speed={_speed_est_m_s:.2f}m/s, "
                    f"method={solution.get('method', '?')}, "
                    f"match_err={solution['error']:.2f}px) — accepting."
                )

            if _is_jump:
                # Precise diagnostics for the WARNING below -- the old message's
                # "err=Npx" is solution['error'], the PROXIMITY MATCH'S OWN
                # reprojection error (match quality), which has nothing to do
                # with why this got flagged as a jump; the actual pos/rot delta
                # that tripped the gate was never logged at all.
                _tv_new = np.asarray(solution["tvec"], np.float64).reshape(3)
                _R_new, _ = cv2.Rodrigues(np.asarray(solution["rvec"], np.float32).reshape(3, 1))
                if _check_vs_prev:
                    _tv_p   = np.asarray(tvec_p, np.float64).reshape(3)
                    _pos_diff_mm = (_tv_new - _tv_p) * 1000.0
                    _R_p,   _ = cv2.Rodrigues(np.asarray(rvec_p, np.float32).reshape(3, 1))
                    # R_new @ R_p.T -- MUST match _pose_jump_too_large's own
                    # convention (this file, _pose_jump_too_large: "R_rel = R_new @
                    # R_ref.T") exactly, not the transposed order. Unlike a pure
                    # magnitude (rotation ANGLE is order-invariant -- conjugate
                    # matrices share a trace/angle), the PER-AXIS breakdown printed
                    # here is NOT order-invariant: R_new@R_p.T and R_p.T@R_new
                    # represent the same physical rotation's axis expressed in two
                    # different frames, so their individual x/y/z components
                    # legitimately differ. Found on a real case: this line's old
                    # (wrong) order printed rot_diff=(19.1,0.7,28.3)deg, all three
                    # individually under the 30deg-per-axis threshold, while the
                    # ACTUAL check (correct order) had rejected the frame -- the
                    # log was describing a different computation than the one that
                    # ran, exactly the "why did this reject, the printed numbers
                    # don't justify it" report this was found from.
                    _rot_diff_deg = np.degrees(np.abs(
                        cv2.Rodrigues((_R_new @ _R_p.T).astype(np.float32))[0].reshape(3)))
                    # 0.15/25.0 mirror _pose_jump_too_large's own scalar-mode
                    # defaults -- only reachable if pose_jump_pos_thresh_m/
                    # _rot_thresh_xyz_deg were both left unset in config (today's
                    # config sets both, so _jump_kw always has these keys in
                    # practice; kept for display accuracy if that ever changes).
                    _pos_thresh = _jump_kw.get('pos_thresh_xyz_m', 0.15)
                    _rot_thresh = _jump_kw.get('rot_thresh_xyz_deg', 25.0)
                    _prev_str = (
                        f"vs prev_pose: pos_diff=({_pos_diff_mm[0]:.0f},{_pos_diff_mm[1]:.0f},{_pos_diff_mm[2]:.0f})mm "
                        f"thresh={tuple(round(v * 1000) for v in _pos_thresh) if isinstance(_pos_thresh, tuple) else f'{_pos_thresh * 1000:.0f}mm scalar'} "
                        f"rot_diff=({_rot_diff_deg[0]:.1f},{_rot_diff_deg[1]:.1f},{_rot_diff_deg[2]:.1f})deg "
                        f"thresh={_rot_thresh if isinstance(_rot_thresh, tuple) else f'{_rot_thresh}deg scalar'}"
                    )
                else:
                    # Re-acquiring: no vs-prev reference was checked at all
                    # (see _check_vs_prev above) -- this reject stands on
                    # predicted_pose alone, filled in by _pred_str below.
                    _prev_str = "vs prev_pose: n/a (re-acquiring)"
                _pred_str = " | vs predicted_pose: n/a (no prediction available -- reject stands on prev_pose alone)"
                if predicted_pose is not None:
                    _tv_pred = np.asarray(predicted_pose[1], np.float64).reshape(3)
                    _pos_vs_pred_mm = float(np.linalg.norm(_tv_new - _tv_pred)) * 1000.0
                    _R_pred, _ = cv2.Rodrigues(np.asarray(predicted_pose[0], np.float32).reshape(3, 1))
                    _rot_vs_pred_xyz_deg = np.degrees(np.abs(
                        cv2.Rodrigues((_R_new @ _R_pred.T).astype(np.float32))[0].reshape(3)))
                    if not _check_vs_prev:
                        _rescued_str = "over threshold"
                    elif _jump_vs_prev:
                        _rescued_str = "also over threshold -- not rescued"
                    else:
                        # NEW case (2026-09-10): vs-prev_pose agreed, but
                        # predicted_pose -- the more authoritative reference
                        # when available -- independently vetoes anyway.
                        _rescued_str = "over threshold -- independent veto despite vs-prev_pose agreement"
                    _pred_str = (f" | vs predicted_pose: dist={_pos_vs_pred_mm:.0f}mm "
                                 f"(vel-dependent thresh={_vel_pos_thresh_m * 1000:.0f}mm @ "
                                 f"speed={_speed_est_m_s:.2f}m/s) "
                                 f"rot_diff=({_rot_vs_pred_xyz_deg[0]:.1f},{_rot_vs_pred_xyz_deg[1]:.1f},{_rot_vs_pred_xyz_deg[2]:.1f})deg "
                                 f"thresh=({_vel_rot_thresh_deg[0]:.1f},{_vel_rot_thresh_deg[1]:.1f},{_vel_rot_thresh_deg[2]:.1f})deg "
                                 f"({_rescued_str})")
                logger.warning(
                    f"[{ctrl_name} | cam {cam_idx} | tracking] Pose jump detected "
                    f"(match_err={solution['error']:.2f}px, method={solution.get('method', '?')}) — "
                    f"{_prev_str}"
                    f"{_pred_str} — attempting brute recovery."
                )
                solution = None
                if allow_expensive_fallback and n_available >= 4:
                    _jump_prior = predicted_pose if predicted_pose is not None else self.prev_pose
                    brute = self.brute_match(blobs, pose_prior=_jump_prior,
                                             other_cameras_blobs=other_cameras_blobs,
                                             blob_mask=blob_mask)
                    if brute is not None:
                        _near_prev = (self.prev_pose is not None and not self._pose_jump_too_large(
                            brute["rvec"], brute["tvec"], self.prev_pose[0], self.prev_pose[1], **_jump_kw
                        ))
                        _near_pred = (predicted_pose is not None and not self._pose_jump_too_large(
                            brute["rvec"], brute["tvec"],
                            predicted_pose[0], predicted_pose[1], **_jump_kw
                        ))
                        if _near_prev or _near_pred:
                            solution = brute

        # ------------------------------------------------------------------
        # Accept / reject
        # ------------------------------------------------------------------

        # Proximity found a solution but error is too high — try brute before giving up
        if (solution is not None and solution["error"] >= _accept_err_px
                and allow_expensive_fallback and n_available >= 4):
            logger.bind(cat="matching_decisions").debug(f"[{ctrl_name} | cam {cam_idx} | track] proximity err={solution['error']:.2f}px exceeds threshold, attempting brute recovery")
            brute = self.brute_match(blobs, pose_prior=self.prev_pose,
                                     other_cameras_blobs=other_cameras_blobs,
                                     blob_mask=blob_mask)
            if brute is not None:
                logger.bind(cat="matching_decisions").debug(f"[{ctrl_name} | cam {cam_idx} | track] brute recovery err={brute['error']:.2f}px")
                solution = brute

        if solution is not None and solution["error"] < _accept_err_px:
            logger.bind(cat="matching_decisions").debug(
                f"[{ctrl_name} | cam {cam_idx} | track] accepted — method={solution.get('method','?')}  "
                f"inliers={len(solution['assignment'])}  err={solution['error']:.2f}px"
            )
            # World-frame controller pose: T_world_ctrl = T_world_cam ∘ T_cam_ctrl
            R_ctrl, _ = cv2.Rodrigues(solution["rvec"])
            T_cam_ctrl = Transform(R_ctrl, solution["tvec"].reshape(3))
            solution["T_world_ctrl"] = self.T_world_cam.compose(T_cam_ctrl)
            return solution

        if solution is not None:
            logger.bind(cat="matching_decisions").debug(f"[{ctrl_name} | cam {cam_idx} | track] rejected — err={solution['error']:.2f}px exceeds {_accept_err_px:.1f}px threshold")
        else:
            logger.bind(cat="matching_decisions").debug(f"[{ctrl_name} | cam {cam_idx} | track] rejected — no solution found")
        return None

    def search(self, blobs: np.ndarray, blob_radii: Optional[np.ndarray] = None,
               blob_brightnesses: Optional[np.ndarray] = None,
               other_cameras_blobs: Optional[List] = None,
               blob_mask: Optional[np.ndarray] = None,
               occluders_per_cam: Optional[Dict] = None,
               allow_expensive_fallback: bool = True) -> Optional[Dict]:
        """Pure pose solve — reads self state, does not commit results.

        Returns a validated solution dict (error below threshold, T_world_ctrl populated)
        or None.  Call apply() with the returned value to commit state changes.

        Thin wrapper: search_cheap() for a candidate; if none and this camera can use
        brute-force (cold-start is unconditional; otherwise gated on
        allow_expensive_fallback — same reasoning as finalize_search), run it
        monolithically; then finalize_search() to validate/accept.
        """
        solution, predicted_pose = self.search_cheap(
            blobs, blob_radii, blob_brightnesses, other_cameras_blobs,
            blob_mask, occluders_per_cam,
        )

        if solution is None:
            n_available = int(blob_mask.sum()) if blob_mask is not None else len(np.asarray(blobs).reshape(-1, 2))
            cam_idx = self.camera.camera_idx
            ctrl_name = self.model.name.replace("_controller", "")
            if self.prev_pose is not None:
                if allow_expensive_fallback and n_available >= 4:
                    logger.bind(cat="matching_decisions").debug(f"[{ctrl_name} | cam {cam_idx} | track] proximity → None, running brute fallback")
                    solution = self.brute_match(blobs, pose_prior=predicted_pose,
                                                other_cameras_blobs=other_cameras_blobs,
                                                blob_mask=blob_mask,
                                                occluders_per_cam=occluders_per_cam)
            else:
                if n_available >= 4:
                    logger.bind(cat="matching_decisions").debug(f"[{ctrl_name} | cam {cam_idx} | track] no prev_pose → cold-start brute")
                    solution = self.brute_match(blobs, other_cameras_blobs=other_cameras_blobs,
                                                blob_mask=blob_mask,
                                                occluders_per_cam=occluders_per_cam)

        return self.finalize_search(
            solution, predicted_pose, blobs, blob_radii, other_cameras_blobs,
            blob_mask, occluders_per_cam, allow_expensive_fallback,
            frame_ts_ns=frame_ts_ns,
        )

    def apply(self, result: Optional[Dict]) -> None:
        """Commit a search() result to tracker state.

        On success: update prev_pose, pose_history, vel_ema, assignment caches.
        On failure: clear transient state so the next frame starts a cold brute search.
        last_good_pose is preserved across failures for re-acquisition plausibility checks.
        """
        if result is not None:
            if self.prev_pose is not None:
                _step = (np.asarray(result["tvec"], np.float64).reshape(3)
                         - np.asarray(self.prev_pose[1], np.float64).reshape(3)).astype(np.float32)
                _beta = float(self._matching_cfg.get("pose_prediction_vel_ema_beta", 0.3))
                self.vel_ema = (_beta * _step + (1.0 - _beta) * self.vel_ema
                               if self.vel_ema is not None else _step)
            elif self.consecutive_failures > 0 and self.last_good_pose is not None:
                _warmstart_max = int(self._matching_cfg.get('pose_gap_velocity_warmstart_frames', 3))
                if self.consecutive_failures <= _warmstart_max:
                    _lg_tvec  = np.asarray(self.last_good_pose[1], np.float64).reshape(3)
                    _new_tvec = np.asarray(result["tvec"],          np.float64).reshape(3)
                    self.vel_ema = ((_new_tvec - _lg_tvec) / self.consecutive_failures).astype(np.float32)
            self.prev_prev_pose  = self.prev_pose
            self.prev_pose       = (result["rvec"], result["tvec"])
            self.pose_history.appendleft((
                np.asarray(result["rvec"], np.float32).reshape(3, 1),
                np.asarray(result["tvec"], np.float32).reshape(3),
            ))
            self.prev_assignment      = result["assignment"]
            self.last_good_pose       = self.prev_pose
            self.last_good_pose_quality = (len(result["assignment"]), float(result.get("error", 0.0)))
            self.last_good_assignment = self.prev_assignment
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.prev_pose       = None
            self.prev_prev_pose  = None
            self.prev_assignment = None
            self.vel_ema         = None
            self.pose_history.clear()


# =========================================================
# 2.5 PER-CONTROLLER TRACKER
# =========================================================

class ControllerTracker:
    """Manages all camera-trackers for a single controller.

    Every available camera searches independently (its own blobs only, no
    aux-camera folding), then all resolved per-camera solutions are fused into
    one pose via fuse_camera_poses(). Owns state propagation to every
    CameraTracker from that fused pose and claimed-blob registration.
    Cross-controller concerns (Voronoi reservation, ordering) remain in TrackingSystem.
    """

    def __init__(self, ctrl_name: str, cameras: Dict[int, "Camera"],
                 trackers: Dict[int, CameraTracker],
                 matching_cfg: Optional[dict] = None,
                 gyro_data: Optional[tuple] = None,
                 accel_data: Optional[tuple] = None,
                 lever_arm: Optional[np.ndarray] = None,
                 g_world_estimator=None,
                 headset_mocap=None,
                 g_world_estimator_abs=None,
                 fusion_cfg: Optional[dict] = None,
                 debug_pose_fusion_cfg: Optional[dict] = None):
        self.ctrl_name        = ctrl_name
        self.cameras          = cameras
        self.trackers         = trackers           # {cam_id: CameraTracker}
        self._matching_cfg    = matching_cfg or {}
        # Informational only (reporting/self-cal anchor from the last successful frame) —
        # every camera searches independently every frame, so this no longer gates work.
        self._designated_primary: Optional[int]  = None

        # Stage 3 IMU integration: (t_ns, values) arrays for this controller's own
        # gyro/accel, already calibrated + axis-corrected + clock-offset-corrected
        # into the vision timestamp domain (see main.py) — or None if unavailable.
        self._gyro_data  = gyro_data
        self._accel_data = accel_data
        # Gravity-consistency diagnostic state (see _log_gravity_consistency):
        # the fused T_world_ctrl.R / timestamp from the last accepted frame.
        self._last_gravity_check_R: Optional[np.ndarray] = None
        self._last_gravity_check_ts: Optional[int] = None

        # Pose-fusion filter — None (fully bypassed, every frame reports the raw
        # vision solve exactly as before this feature existed) unless fusion_cfg
        # explicitly enables it. fusion.filter_type picks the implementation:
        # "kalman" (default, src/pose_fusion.py, covariance/chi2-gate based) or
        # "heuristic" (src/pose_fusion_heuristic.py, explicit ordered rules +
        # self-calibrating thresholds — see /home/nikitakarpuks/.claude/plans/
        # heuristic-direction-further-write-tingly-moonbeam.md for why). Both
        # share the exact same public interface, so nothing below this branch
        # needs to know or care which one got constructed.
        self._g_world_estimator = g_world_estimator
        # Headset-ego-motion correction for dead-reckoning (see src.mocap_data.world_pose /
        # src.imu_data.predict_headset_relative_pose) -- both None unless mocap.enabled and a
        # headset mocap trajectory actually loaded (main.py); HeuristicPoseFusionFilter treats
        # either being None as "correction unavailable, use the uncorrected path" (fail-open).
        self._headset_mocap = headset_mocap
        self._g_world_estimator_abs = g_world_estimator_abs
        _filter_cls = (HeuristicPoseFusionFilter
                       if (fusion_cfg or {}).get("filter_type", "kalman") == "heuristic"
                       else PoseFusionFilter)
        _filter_kwargs = (
            {"ctrl_name": self.ctrl_name, "headset_mocap": headset_mocap,
             "g_world_estimator_abs": g_world_estimator_abs}
            if _filter_cls is HeuristicPoseFusionFilter else {}
        )
        self._fusion_filter: Optional[Union[PoseFusionFilter, HeuristicPoseFusionFilter]] = (
            _filter_cls(gyro_data, accel_data, lever_arm, g_world_estimator, fusion_cfg or {}, **_filter_kwargs)
            if (fusion_cfg or {}).get("enabled", False) else None
        )
        # visualization.pose_fusion_debug (config/config.yml) — gates the extra
        # per-frame cost of capturing PoseFusionFilter.debug_snapshot()/predict_dense()
        # for the rerun debug tool below; a no-op when disabled (the default) or
        # when self._fusion_filter is None regardless of this flag.
        self._debug_pose_fusion_cfg = debug_pose_fusion_cfg or {}
        self._debug_pose_fusion     = self._debug_pose_fusion_cfg.get("enabled", False)

        # Other enabled controllers' own ControllerTracker instances -- wired post-
        # construction via set_sibling_controllers() (TrackingSystem builds every
        # ControllerTracker before any of them can reference the others), same
        # pattern/site as PoseFusionFilter.set_siblings(). Used by
        # _commit_fused_solution's escape hatch to also reset a sibling's own
        # filter/camera-tracker state when THIS controller's rejection is an
        # abs_reject-flagged identity-swap signature (see there for why a swap
        # implicates both controllers' recent trust, not just this one's).
        self._sibling_controllers: list = []

        # True from the first frame vision ever produces a real candidate for
        # this controller (set in _commit_fused_solution, unconditionally --
        # Guards _mark_all_lost's per-frame IMU-only-propagation decision
        # (matching.imu_only_propagation_max_s) against being made twice on
        # the SAME real frame -- _mark_all_lost is called once per FAILED
        # search ATTEMPT, not once per frame: main.py retries a controller
        # whose cheap search found nothing with an immediate cold re-detect +
        # brute-force attempt (same frame_ts_ns), and if that also fails,
        # ControllerTracker.update() calls _mark_all_lost a second time for
        # it (controller.py:1114/1349, pre-existing pattern, unrelated to
        # this feature). Without this guard, a same-frame re-entry could
        # double-propagate pose_history for one real frame. The budget itself
        # needs no separate counter/reset any more -- it reads real elapsed
        # time off self._fusion_filter.last_update_ts_ns, which already
        # advances on every real accept (_commit_fused_solution), so a new
        # loss streak automatically gets a fresh budget with no bookkeeping
        # here.
        self._last_imu_propagate_ts_ns: Optional[int] = None

        # The IMU-only pose _mark_all_lost actually decided on THIS frame (a
        # Transform if within budget and IMU coverage allowed it, None
        # otherwise) -- see imu_only_predicted_pose. Single source of truth
        # so the 3D display can never show more consecutive IMU-only frames
        # than the search-anchor propagation itself was budgeted for.
        self._last_imu_only_pose: Optional[Transform] = None

    def set_sibling_controllers(self, siblings: list) -> None:
        self._sibling_controllers = list(siblings)

    def imu_only_predicted_pose(self, frame_ts_ns: int) -> Optional[Transform]:
        """IMU-only dead-reckoned world pose for THIS frame, or None if
        _mark_all_lost didn't propagate one (no fusion filter, no prior state
        yet, no IMU coverage, g_world not converged, or the
        imu_only_propagation_max_s budget for this loss streak -- real
        elapsed time since the last accepted vision update -- is already
        spent). Used by main.py to report/display a pose on a frame with
        zero vision candidates, instead of hiding the controller outright.

        Deliberately reads _mark_all_lost's own cached decision
        (_last_imu_only_pose) rather than calling self._fusion_filter.predict()
        again here -- an earlier version did call it independently, which
        meant the display had NO cap at all (predict() itself has no
        duration limit): it kept extrapolating and showing a pose for as
        long as IMU coverage/g_world allowed, completely ignoring the
        budget the search-anchor side was honoring (found in review: 25
        consecutive IMU-only frames shown despite a configured cap of
        ~0.066s). Reusing the same cached Transform guarantees the display
        can never show more IMU-only time than the search anchor was
        actually budgeted for."""
        return self._last_imu_only_pose

    def force_cold_start(self, reason: str = "") -> None:
        """Immediately clear this controller's fusion-filter state and every
        camera's warm-start prior -- the same reset shape as
        should_force_cold_start's own escape hatch (_commit_fused_solution)
        or a grace-exhausted vision loss (_mark_all_lost), but for a caller
        that has independently decided this controller's CURRENT belief is
        provably wrong and must not be reported or warm-started from again.

        Used by main.py's cross-controller collision check: an IMU-only
        prediction (no real vision this frame) that lands on top of a
        DIFFERENT controller's actual vision-confirmed position this frame is
        proof that prediction is wrong (two rigid controllers cannot share a
        tracked origin) -- resetting here means the very next frame starts a
        genuine cold re-detect instead of continuing to coast from (and
        potentially re-warm-start off) the disproven position.

        Also clears last_good_pose/last_good_pose_ts_ns (clear_prior() itself
        deliberately does NOT -- it's meant to survive an ordinary vision
        loss as a re-acquisition reference, see CameraTracker.clear_prior's
        own docstring). But this reset means the recent belief was PROVEN
        wrong, not just lost -- last_good_pose was almost certainly captured
        during the same contaminated (misidentified) stretch, so leaving it
        alive would hand CameraTracker.finalize_search's own cold-start
        plausibility check a corrupted reference to judge the NEXT real
        candidate against, right after we just established it can't be
        trusted."""
        if self._fusion_filter is not None:
            self._fusion_filter.reset()
        for _tracker in self.trackers.values():
            _tracker.clear_prior()
            _tracker.last_good_pose = None
            _tracker.last_good_pose_ts_ns = None
            _tracker.last_good_pose_quality = None
        # No separate IMU-only-propagation counter to reset -- self._fusion_
        # filter.reset() above already nulls last_update_ts_ns (and R/p), so
        # _mark_all_lost's elapsed-time budget check finds no anchor to
        # measure from and naturally grants no IMU-only propagation until a
        # real vision update lands again.
        self._last_imu_only_pose = None
        if reason:
            logger.bind(cat="matching_decisions").info(f"[{self.ctrl_name}] FORCE COLD-START: {reason}")

    def debug_fusion_state(self, frame_ts_ns: int):
        """Pose-fusion debug snapshot + dense IMU path for a frame where this
        controller had NO vision solution at all (cam_solutions empty this
        frame -- update() returns None, _commit_fused_solution never runs, so
        it never gets a chance to capture these itself). Callers should use
        this from the "no solution this frame" branch so the rerun debug
        tool's IMU-predicted-path curve keeps growing across a real full-
        occlusion gap too, not just a reject streak where vision keeps
        finding candidates but the filter keeps rejecting them (found in
        review). Returns (fusion_debug, fusion_imu_path), both None if
        debug logging or the filter itself isn't enabled."""
        if self._fusion_filter is None or not self._debug_pose_fusion:
            return None, None
        fusion_debug = self._fusion_filter.debug_snapshot(frame_ts_ns)
        stride = int(self._debug_pose_fusion_cfg.get("path_stride", 4))
        fusion_imu_path = self._fusion_filter.predict_dense(frame_ts_ns, sample_every_n=stride)
        _imu_x = float(fusion_imu_path[1][-1][0]) if fusion_imu_path is not None and len(fusion_imu_path[0]) else None
        logger.bind(cat="pose_fusion").debug(
            f"[{self.ctrl_name}] LOST frame debug_fusion_state ts={frame_ts_ns} "
            f"has_state={self._fusion_filter.R is not None} "
            f"trust={fusion_debug.get('trust')} imu_path={'yes' if fusion_imu_path is not None else 'None'} imu_x={_imu_x}"
        )
        return fusion_debug, fusion_imu_path

    def _mark_all_lost(self, frame_ts_ns: int) -> None:
        """Record a failed frame on every camera-tracker and, once the
        configured grace period is exhausted, clear prev_pose/pose_history so
        the controller stops warm-starting off stale history. A single retry
        (the default grace of 1) still gets to warm-start next frame per the
        tracking_lost_last_frame contract in CameraTracker.finalize_search;
        beyond that, surviving history was letting a controller with no real
        track (or one truly gone cold) look "warm" forever — see
        get_predicted_led_projections_per_camera / ctrl_has_prior in main.py,
        which key off pose_history alone and don't otherwise expire it.
        Clearing here also starves _build_extrapolated_occluders (no
        extrapolated pose survives), so a genuinely cold controller no longer
        casts a phantom cross-controller occlusion mask.

        Also checks self._fusion_filter.should_force_cold_start(frame_ts_ns) on
        every call (plan Phase 5's escape hatch: elapsed-time-since-last-accepted-
        update past `max_coast_s`, or too many consecutive live rejects) and
        resets it if so — DELIBERATELY NOT tied to the vision-side grace above.
        An earlier version reset the filter on that same 1-frame grace, which
        turned out wrong: this method runs every frame during ANY vision loss,
        including the short (sub-second) full-occlusion gaps the swap-bug
        mechanism actually depends on (confirmed against this recording's real
        gap: ~0.5s, both controllers, well inside the ~1s dead-reckoning-
        reliable horizon) — clearing the filter's state after just 2 failed
        frames would erase the exact pre-loss prediction that must survive to
        catch the swap on reacquisition. should_force_cold_start's own
        elapsed-time criterion is what should gate this, not vision's frame-
        count grace, which exists for an unrelated reason (stop warm-starting
        the pixel-space search off a stale prediction, an ~ms-scale concern).

        IMU-only propagation (matching.imu_only_propagation_max_s, default
        0.066s): before falling back to the grace-frame logic above, try
        dead-reckoning a fresh world pose from self._fusion_filter.predict()
        and, if IMU coverage/g_world allow it, warm-start every camera's
        pose_history from THAT instead of leaving it frozen at the last
        vision-confirmed pose. This keeps the search neighbourhood centered
        near where the controller actually is during a short vision gap
        (occlusion, a missed detection, a rejected re-acquisition) rather
        than stale, for up to this much REAL ELAPSED TIME since the fusion
        filter's last accepted vision update -- "resets" the instant vision
        produces a real candidate again (_commit_fused_solution advances
        self._fusion_filter.last_update_ts_ns itself; no separate counter to
        reset). Past the budget (or whenever predict() itself fails-open --
        no IMU coverage yet, g_world not converged, fusion disabled), this
        degrades to exactly the old grace-frame-then-clear behavior; short-
        horizon accel+gyro dead-reckoning is validated for gaps well beyond
        this budget (see HeuristicPoseFusionFilter/PoseFusionFilter's own
        docstrings), so a fraction of a second of it is comfortably inside
        the reliable regime.

        TIME-based (2026-09-09), not frame-counted -- was
        imu_only_propagation_max_frames (a per-frame counter incremented
        once per call here) until this recording's own oscillating frame
        rate (alternates ~11ms/~22ms) was found producing a real
        discrepancy against CameraTracker.finalize_search's own cold-start
        widening check, which already used real elapsed time
        (last_good_pose_ts_ns) -- a frame-counted budget here and a
        time-based tolerance there could disagree about "how stale is this"
        for the exact same loss streak, and the two numbers weren't even in
        comparable units in the resulting log line. Using
        self._fusion_filter.last_update_ts_ns as the elapsed-time anchor
        (instead of a separately-tracked "started this loss streak"
        timestamp) is deliberate: it's the exact same event
        last_good_pose_ts_ns is set from (both updated together in
        _commit_fused_solution's `if accepted:` block), so both mechanisms
        now read off one identical clock with no unit or anchor mismatch
        possible.
        """
        _grace = int(self._matching_cfg.get('tracking_lost_grace_frames', 1))
        _imu_only_max_s = float(self._matching_cfg.get('imu_only_propagation_max_s', 0.066))

        _elapsed_since_real_update_s = None
        if self._fusion_filter is not None and self._fusion_filter.last_update_ts_ns is not None:
            _elapsed_since_real_update_s = max(
                0.0, (frame_ts_ns - self._fusion_filter.last_update_ts_ns) / 1e9)

        _imu_pose = None
        _already_handled_this_frame = (frame_ts_ns == self._last_imu_propagate_ts_ns)
        if not _already_handled_this_frame:
            if self._fusion_filter is not None:
                # Bump frames_since_update on EVERY real lost frame, unconditionally --
                # NOT just while still inside the IMU-only coast's own time budget below.
                # Cheap (a timestamp-deduped increment, no dead-reckoning math), unlike
                # predict() itself, so this doesn't reintroduce the growing-window cost
                # the budget exists to cap. See HeuristicPoseFusionFilter.note_real_frame's
                # own docstring for the real false-reject this fixes: without it,
                # frames_since_update silently freezes the moment the budget below is
                # exhausted, keeping imu_frame_scale (and the hard implausibility gate it
                # guards) artificially undecayed for the rest of an arbitrarily long loss.
                self._fusion_filter.note_real_frame(frame_ts_ns)
            if (self._fusion_filter is not None and _elapsed_since_real_update_s is not None
                    and _elapsed_since_real_update_s < _imu_only_max_s
                    and self._fusion_filter.velocity_established):
                # velocity_established gate: predict()'s position component is
                # dead-reckoned from self._fusion_filter.v, which right after a
                # bootstrap/reset/fail-open is a fabricated v=0, not a real
                # measurement (see HeuristicPoseFusionFilter.velocity_established's
                # own comment) -- warm-starting the next search's neighborhood
                # from that position has no more basis than just leaving prev_pose
                # alone, so this falls through to the grace-frame-then-clear path
                # below instead of fabricating an anchor to search around.
                _predicted = self._fusion_filter.predict(frame_ts_ns)
                if _predicted is not None:
                    _imu_pose = Transform(*_predicted)
            self._last_imu_propagate_ts_ns = frame_ts_ns
            # Cache THIS frame's decision for imu_only_predicted_pose (the 3D
            # display) to read -- unconditionally on every NEW frame (not
            # nested inside the budget check above), so it correctly clears
            # to None the moment the budget is exhausted (or IMU coverage/
            # g_world isn't there), rather than freezing at its last non-None
            # value forever once the inner condition stops firing. A same-
            # frame re-entry (_already_handled_this_frame) leaves it alone --
            # that decision was already made and cached on the first call.
            self._last_imu_only_pose = _imu_pose

        if _imu_pose is not None:
            self._propagate_pose_history(_imu_pose, frame_ts_ns)
            if self._debug_pose_fusion:
                logger.bind(cat="pose_fusion").debug(
                    f"[{self.ctrl_name}] IMU-only propagation "
                    f"({_elapsed_since_real_update_s:.3f}s/{_imu_only_max_s:.3f}s): "
                    f"keeping search anchor warm during vision loss, pos={_imu_pose.t}"
                )

        for _t in self.trackers.values():
            _t.tracking_lost_last_frame = True
            _t.consecutive_failures += 1
            if _imu_pose is None and _t.consecutive_failures > _grace:
                _t.clear_prior()
        if self._fusion_filter is not None and self._fusion_filter.should_force_cold_start(frame_ts_ns):
            if self._debug_pose_fusion:
                _elapsed = ((frame_ts_ns - self._fusion_filter.last_update_ts_ns) / 1e9
                            if self._fusion_filter.last_update_ts_ns is not None else None)
                logger.bind(cat="pose_fusion").debug(
                    f"[{self.ctrl_name}] RESET during vision loss: ts={frame_ts_ns} elapsed_s={_elapsed} "
                    f"consecutive_rejects={self._fusion_filter.consecutive_rejects}"
                )
            self._fusion_filter.reset()

    def update(
        self,
        avail:         Dict[int, tuple],        # cam_id → (blobs, radii, brts, orig_idx)
        obs_src:       Dict[int, np.ndarray],   # full observations per camera
        rad_src:       Dict,
        brt_src:       Dict,
        claimed_blobs: Dict[int, Set[int]],     # mutated in place
        frame_ts_ns:   int,
        fixed_primary_cam: Optional[int] = None,
        occluders_per_cam: Optional[Dict] = None,
        self_cal=None,
        pool=None,
        allow_brute: bool = True,
        force_brute: bool = False,
    ) -> Optional[Dict]:
        """allow_brute=False stops after the cheap (proximity/prior_constrained)
        pass and returns None instead of running tier-round brute-force recovery
        — lets a caller detect "every camera's cheap search failed" and react
        (e.g. re-detect blobs cold) before paying for brute-force on blobs that
        were detected under a now-untrustworthy prediction.

        force_brute=True skips the cheap pass entirely and goes straight to
        brute-force, with no pose_prior (same as a cold-start first frame) —
        for a caller that already re-detected blobs cold after cheap search
        failed on the (now known bad) extrapolated pose: retrying cheap search
        against that same untrustworthy prediction, just with a fuller blob
        set, would fail for the same reason it failed the first time."""
        if not avail:
            self._mark_all_lost(frame_ts_ns)
            return None

        # ── Every available camera searches independently ──────────────────────
        # Own blobs only — no aux-camera folding at the per-camera search level;
        # cross-camera fusion happens once, below, via fuse_camera_poses().
        #
        # Pass 1: cheap methods (proximity + prior_constrained) for every camera — a
        # camera with a prior whose cheap methods come up short can simply contribute
        # nothing this frame and get warm-started next frame from the fused pose, at
        # zero cost. Only if NO camera produced anything this frame (the controller is
        # about to lose track entirely) do we pay for brute-force recovery — run in
        # tier-rounds across every camera needing it (see below), rather than one
        # camera exhausting its whole tier ladder before the next is even tried.
        #
        # When `pool` is given, both stages dispatch to it: pass 1 submits one cheap-
        # search task per camera; the tier-round loop submits one brute-tier task per
        # camera still needing that round, waiting for the whole round before
        # deciding to widen or stop (no mid-tier cancellation — see plan). Without a
        # pool, both stages fall back to the equivalent sequential loop.
        cam_solutions: List[dict] = []
        predicted_pose_by_cid: Dict[int, Optional[Tuple[np.ndarray, np.ndarray]]] = {}

        t_cheap = 0.0
        if not force_brute:
            _t_cheap_start = time.perf_counter()
            _cheap_specs = []   # (cid, tracker, obs_full, rad_full, brt_full, mask, av_orig)
            for cid, (av_blobs, av_radii, av_brts, av_orig) in avail.items():
                if len(av_blobs) == 0:
                    continue
                tracker  = self.trackers[cid]
                obs_full = obs_src[cid]
                rad_full = rad_src.get(cid) if rad_src else None
                brt_full = brt_src.get(cid) if brt_src else None
                mask = np.zeros(len(obs_full), dtype=bool)
                mask[av_orig] = True
                _cheap_specs.append((cid, tracker, obs_full, rad_full, brt_full, mask, av_orig))

            _cheap_raw: Dict[int, tuple] = {}   # cid -> (solution, predicted_pose)
            # Computed ONCE per controller per frame, not once per camera --
            # see _predicted_world_for_vel_ema's own docstring.
            _predicted_world = _predicted_world_for_vel_ema(
                self._fusion_filter, self.trackers.values(), frame_ts_ns)
            if pool is not None and _cheap_specs:
                from src.parallel_search import run_cheap_search
                _futures = {}
                for cid, tracker, obs_full, rad_full, brt_full, mask, av_orig in _cheap_specs:
                    prior = {
                        'prev_pose':       tracker.prev_pose,
                        'prev_prev_pose':  tracker.prev_prev_pose,
                        'pose_history':    tracker.pose_history,
                        'vel_ema':         _vel_ema_with_imu_fallback(
                            tracker, _predicted_world, self.cameras[cid], frame_ts_ns),
                        'prev_assignment': tracker.prev_assignment,
                        'gyro_rel_R':      _gyro_rel_R_for(self._gyro_data, tracker.pose_history, frame_ts_ns),
                    }
                    _futures[cid] = pool.submit(
                        run_cheap_search, (self.ctrl_name, cid), self._matching_cfg, prior,
                        obs_full, frame_ts_ns, rad_full, brt_full, mask, occluders_per_cam,
                    )
                for cid, fut in _futures.items():
                    solution, predicted_pose, norm_prev, norm_prev_prev = fut.result()
                    self.trackers[cid].prev_pose      = norm_prev
                    self.trackers[cid].prev_prev_pose = norm_prev_prev
                    _cheap_raw[cid] = (solution, predicted_pose)
            else:
                for cid, tracker, obs_full, rad_full, brt_full, mask, av_orig in _cheap_specs:
                    _cheap_raw[cid] = tracker.search_cheap(
                        obs_full, frame_ts_ns, blob_radii=rad_full, blob_brightnesses=brt_full,
                        other_cameras_blobs=None, blob_mask=mask,
                        occluders_per_cam=occluders_per_cam,
                        gyro_rel_R=_gyro_rel_R_for(self._gyro_data, tracker.pose_history, frame_ts_ns),
                        vel_ema_override=_vel_ema_with_imu_fallback(
                            tracker, _predicted_world, self.cameras[cid], frame_ts_ns),
                    )

            for cid, tracker, obs_full, rad_full, brt_full, mask, av_orig in _cheap_specs:
                solution, predicted_pose = _cheap_raw[cid]
                predicted_pose_by_cid[cid] = predicted_pose
                solution = tracker.finalize_search(
                    solution, predicted_pose, obs_full,
                    blob_radii=rad_full, other_cameras_blobs=None,
                    blob_mask=mask, occluders_per_cam=occluders_per_cam,
                    allow_expensive_fallback=False, frame_ts_ns=frame_ts_ns,
                )
                if solution is None:
                    continue
                if not solution.get('_orig_idx', False):
                    solution["assignment"] = [(av_orig[b], lid) for b, lid in solution["assignment"]]
                cam_solutions.append({"cam_id": cid, "tracker": tracker, "solution": solution})

            t_cheap = time.perf_counter() - _t_cheap_start

        t_new_state = t_tier_rounds = t_finalize = 0.0
        n_rounds = 0
        _pool_rt_ms: Dict[int, float] = {}   # cid -> worst brute-tier pool round-trip, if pool was used

        # ── Weak single-camera cheap accept + another eligible camera failed ────
        # "Only if NO camera produced anything this frame do we pay for brute-force
        # recovery" (see this method's own docstring) is too coarse: it treats a
        # single low-inlier proximity accept from one camera as equally trustworthy
        # as a solid multi-camera one, even when ANOTHER camera that had real
        # detected blobs this frame (in _cheap_specs, not just geometrically
        # in-frustum) came up with nothing or got jump-rejected. Confirmed on a
        # real recording (frame_range 800-900 relative frames 50/51/53): cam1 kept
        # accepting 4-7-inlier proximity matches with several of its own detected
        # blobs left unmatched, while cam0 -- which had its own real candidate
        # blobs every one of those frames -- never got to independently confirm or
        # contradict it, letting a single weak, partially-wrong camera lock in the
        # whole controller's fused pose frame after frame with no cross-check.
        #
        # "Weak" = either too few inliers (reuses strong_match_inliers, the SAME
        # "is this actually a strong match" floor PoseSearcher's own brute
        # strong_found early-exit uses -- not a second, separate threshold) OR too
        # low a blob-utilization ratio (inliers / this camera's own available blob
        # count this frame) -- inlier count alone missed frame 50's case (7
        # inliers, comfortably >= strong_match_inliers, but from only 9 of the
        # camera's own 13 detected blobs, the rest genuinely unmatched).
        # weak_solo_blob_utilization_floor=0.7 is empirically set: surveyed 1684
        # real accepted proximity solutions (walk_medium, 3000 frames) and only
        # 4.04% ever fall below 0.7 (median ratio is 1.0, i.e. most accepts use
        # every detected blob) -- low enough that this essentially never fires on
        # a normal, healthy single-camera accept.
        _weak_solo_cids: List[int] = []
        if not force_brute and cam_solutions:
            _av_count_by_cid = {cid: len(av_orig) for cid, _t, _o, _r, _b, _m, av_orig in _cheap_specs}
            _weak_solo_cids = _weak_solo_accept_cids(
                cam_solutions, [c[0] for c in _cheap_specs], _av_count_by_cid, self._matching_cfg)
            if _weak_solo_cids and not allow_brute:
                # Can't run the cross-check brute-force here -- this is the FIRST
                # sequential fallback stage (main.py's _update_ctrl(allow_brute=False)
                # after update_warm_batch itself deferred, see that method's own
                # matching comment). Discard the weak solution and signal failure
                # (empty cam_solutions -> "if not cam_solutions" below) so the
                # caller's NEXT stage (cold re-detect + force_brute=True, the same
                # path any other total cheap-search failure already takes) gets a
                # real, unrestricted brute-force attempt instead of this method
                # quietly returning a weak, single-camera, un-cross-checked solution.
                logger.bind(cat="matching_decisions").debug(
                    f"[{self.ctrl_name}] weak solo accept (cam(s) "
                    f"{sorted({cs['cam_id'] for cs in cam_solutions})}) while cam(s) "
                    f"{_weak_solo_cids} had blobs but no accepted solution -- "
                    f"allow_brute=False here, deferring to cold re-detect + brute"
                )
                cam_solutions = []
                _weak_solo_cids = []
            elif _weak_solo_cids:
                logger.bind(cat="matching_decisions").debug(
                    f"[{self.ctrl_name}] weak solo accept (cam(s) "
                    f"{sorted({cs['cam_id'] for cs in cam_solutions})}) while cam(s) "
                    f"{_weak_solo_cids} had blobs but no accepted solution -- "
                    f"forcing brute-force cross-check on {_weak_solo_cids}"
                )

        if force_brute or (not cam_solutions and allow_brute) or (allow_brute and _weak_solo_cids):
            # ── Tier-round brute-force recovery ─────────────────────────────────
            # For a plain weak-solo trigger (not force_brute, not the "nothing at
            # all" case), only the camera(s) that failed cheap search actually run
            # brute-force below -- the already-accepted weak solution stays in
            # cam_solutions untouched (never re-run/duplicated), and a real brute
            # candidate from the other camera, if found, is simply APPENDED
            # alongside it for joint_fusion/conflict-resolution to weigh together.
            _brute_cids = (
                list(avail.keys()) if (force_brute or not cam_solutions) else _weak_solo_cids
            )
            # tier_0 for every camera with a chance (>=4 available blobs); wait for
            # the whole round to finish before deciding anything (no mid-tier
            # cancellation); stop widening the instant any camera hits strong_found
            # this round.
            states: Dict[int, object] = {}
            _other_cams_by_cid: Dict[int, List[Tuple]] = {
                cid: _build_other_cameras_blobs(self.cameras, obs_src, cid) for cid in avail
            }
            _t_new_state_start = time.perf_counter()
            _confirm_frames = int(self._matching_cfg.get('cold_brute_force_confirm_frames', 3))
            for cid in _brute_cids:
                av_blobs, av_radii, av_brts, av_orig = avail[cid]
                tracker = self.trackers[cid]
                # force_brute=True covers TWO distinct cases (main.py:585 and
                # main.py:633): a truly cold controller (no prior anywhere), and
                # a controller that was warm THIS frame whose cheap search just
                # failed, falling back to an immediate cold re-detect + brute
                # attempt. Only the former should pay the persistence-streak
                # cost below — a just-lost controller's prior is still trusted
                # (that's the whole point of tracking_lost_grace_frames) and
                # deserves an immediate recovery attempt, not an added
                # multi-frame confirmation delay. _mark_all_lost only clears
                # prev_pose once the grace period is exhausted, so
                # prev_pose is None is exactly "truly cold" here; prev_pose is
                # still set means "recently lost, within grace" — skip the gate.
                #
                # self._ever_tracked used to ALSO unconditionally exempt a
                # controller confirmed real earlier this session, even once
                # prev_pose had been cleared (grace period long expired) --
                # reasoning: "might not even be a real controller yet" (frame
                # 1) doesn't apply once we already know it's real, so don't
                # pay the multi-frame confirmation delay again (confirmed on
                # this project's own recording: frames 25-26 lost purely to
                # this gate). REMOVED 2026-09-11: that exemption disables the
                # streak check FOREVER after a controller's first successful
                # track, not just briefly after loss -- the persistence
                # streak isn't only about "is this a real controller," it's
                # about "is THIS PARTICULAR reacquisition candidate real,"
                # which stays just as necessary for an already-known-real
                # controller's Nth reacquisition as its 1st. Confirmed on a
                # real case (frame_range 1750-1800 relative frame 48): a
                # previously-tracked, long-lost right_controller re-acquired
                # on the very first frame with exactly min_inliers=4 blobs
                # available -- a zero-redundancy accept (every available blob
                # instantly counted as an inlier, RANSAC had nothing to reject
                # against) that immediately lost warm proximity again next
                # frame and forced a collision-based RESET two frames later.
                # The frames-25-26 case above is still covered without this
                # exemption: prev_pose stays set through tracking_lost_grace_
                # frames (see the comment above), so a controller that
                # re-solves within its own grace window never reaches this
                # branch at all regardless of _ever_tracked -- only a gap
                # that outlasts the grace period now also pays the same
                # streak-confirmation cost a first-time acquisition does,
                # which is the intended behavior per cold_brute_force_
                # confirm_frames' own config.yml description.
                if force_brute and tracker.prev_pose is None:
                    # Persistence gate: require _confirm_frames CONSECUTIVE
                    # frames of >= min_inliers blobs before paying for a full
                    # P3P/gate enumeration, since real reacquired LEDs persist
                    # frame-to-frame while noise blobs don't (e.g. blob counts
                    # 4,5,1 never trigger; 4,5,6 triggers on the 3rd frame).
                    # Reset on success lives in _fuse_and_finalize alongside
                    # consecutive_failures.
                    if len(av_blobs) >= tracker._pose_searcher._c_brute_min_inliers:
                        tracker._consecutive_good_blob_frames += 1
                    else:
                        tracker._consecutive_good_blob_frames = 0
                    if tracker._consecutive_good_blob_frames < _confirm_frames:
                        logger.bind(cat="matching_decisions").debug(
                            f"[{self.ctrl_name} | cam {cid}] cold brute-force gated: "
                            f"streak={tracker._consecutive_good_blob_frames}/{_confirm_frames} "
                            f"(n_blobs={len(av_blobs)})"
                        )
                        continue
                if len(av_blobs) == 0:
                    continue
                obs_full = obs_src[cid]
                mask = np.zeros(len(obs_full), dtype=bool)
                mask[av_orig] = True
                states[cid] = tracker._pose_searcher.new_brute_state(
                    obs_full, pose_prior=predicted_pose_by_cid.get(cid),
                    other_cameras_blobs=_other_cams_by_cid[cid],
                    blob_mask=mask, occluders_per_cam=occluders_per_cam,
                )
            t_new_state = time.perf_counter() - _t_new_state_start

            _any_ps = next(
                (self.trackers[cid]._pose_searcher for cid, st in states.items() if st is not None),
                None,
            )
            max_tiers = len(_any_ps._c_brute_depth_tiers) if _any_ps is not None else 0

            _t_tier_start = time.perf_counter()
            if pool is not None:
                from src.parallel_search import run_brute_tier
                for tier_idx in range(max_tiers):
                    remaining = [cid for cid, st in states.items() if st is not None and not st.strong_found]
                    if not remaining:
                        break
                    n_rounds += 1
                    _submit_t0 = time.perf_counter()
                    _futures = {
                        cid: pool.submit(run_brute_tier, (self.ctrl_name, cid), states[cid], tier_idx)
                        for cid in remaining
                    }
                    for cid, fut in _futures.items():
                        states[cid] = fut.result()
                        _rt = (time.perf_counter() - _submit_t0) * 1000
                        _pool_rt_ms[cid] = max(_pool_rt_ms.get(cid, 0.0), _rt)
                    if any(states[cid].strong_found for cid in remaining):
                        break
            else:
                for tier_idx in range(max_tiers):
                    remaining = [cid for cid, st in states.items() if st is not None and not st.strong_found]
                    if not remaining:
                        break
                    n_rounds += 1
                    for cid in remaining:
                        self.trackers[cid]._pose_searcher.brute_search_tier(states[cid], tier_idx)
                    if any(states[cid].strong_found for cid in remaining):
                        break
            t_tier_rounds = time.perf_counter() - _t_tier_start

            _t_finalize_start = time.perf_counter()
            for cid, st in states.items():
                if st is None:
                    continue
                tracker = self.trackers[cid]
                sol = tracker._pose_searcher.finalize_brute_state(st)
                if sol is None:
                    continue
                _, _, _, av_orig = avail[cid]
                if not sol.get('_orig_idx', False):
                    sol["assignment"] = [(av_orig[b], lid) for b, lid in sol["assignment"]]
                obs_full = obs_src[cid]
                rad_full = rad_src.get(cid) if rad_src else None
                mask = np.zeros(len(obs_full), dtype=bool)
                mask[av_orig] = True
                sol = tracker.finalize_search(
                    sol, predicted_pose_by_cid.get(cid), obs_full,
                    blob_radii=rad_full, other_cameras_blobs=_other_cams_by_cid[cid],
                    blob_mask=mask, occluders_per_cam=occluders_per_cam,
                    allow_expensive_fallback=True, frame_ts_ns=frame_ts_ns,
                )
                if sol is not None:
                    cam_solutions.append({"cam_id": cid, "tracker": tracker, "solution": sol})
            t_finalize = time.perf_counter() - _t_finalize_start

        _pool_rt_str = (
            "  pool_rt=[" + " ".join(f"cam{c}={ms:.1f}ms" for c, ms in sorted(_pool_rt_ms.items())) + "]"
            if _pool_rt_ms else ""
        )
        logger.bind(cat="timings").debug(
            f"[{self.ctrl_name}] update timing: cheap={t_cheap*1000:.1f}ms  "
            f"brute(new_state={t_new_state*1000:.1f}ms "
            f"tier_rounds={t_tier_rounds*1000:.1f}ms[{n_rounds} rounds] "
            f"finalize={t_finalize*1000:.1f}ms){_pool_rt_str}"
        )

        if not cam_solutions:
            self._mark_all_lost(frame_ts_ns)
            return None

        return self._fuse_and_finalize(
            cam_solutions, obs_src, claimed_blobs, frame_ts_ns,
            fixed_primary_cam=fixed_primary_cam, self_cal=self_cal,
        )

    def _compute_fused_solution(
        self,
        cam_solutions: List[dict],
        obs_src: Dict[int, np.ndarray],
        fixed_primary_cam: Optional[int] = None,
    ) -> Dict:
        """Fuse every camera's independent solution into one controller pose
        and build the resulting solution dict — pure, no tracker-state
        mutation, no claimed_blobs registration, no self-cal feed. Split out
        of _fuse_and_finalize so a caller with several simultaneously-solved
        candidates for DIFFERENT controllers (TrackingSystem.update_cold_batch)
        can compare them — by solution["error"] and matched-pair counts — and
        resolve cross-controller conflicts before committing any one of them
        via _commit_fused_solution. cam_solutions must be non-empty.
        """
        from src.pose_search import fuse_camera_poses

        # ── Fuse every camera's independent solution into one pose ─────────────
        _fuse_in = [
            {
                "camera":       self.cameras[cs["cam_id"]],
                "blobs":        obs_src[cs["cam_id"]],
                "pairs":        cs["solution"]["assignment"],
                "T_world_ctrl": cs["solution"]["T_world_ctrl"],
                "error":        cs["solution"]["error"],
                "confidence":   cs["solution"].get("confidence", 1.0),
            }
            for cs in cam_solutions
        ]
        model_positions = next(iter(self.trackers.values())).model.positions
        _t_fuse_start = time.perf_counter()
        T_world_ctrl, fused_err = fuse_camera_poses(
            _fuse_in, model_positions, matching_cfg=self._matching_cfg,
        )
        _t_fuse = time.perf_counter() - _t_fuse_start

        # ── Anchor camera for reporting/self-cal: fixed_primary_cam if it solved
        # this frame, else whichever camera had the most inlier pairs ──────────
        _solved_ids = {cs["cam_id"] for cs in cam_solutions}
        if fixed_primary_cam is not None and fixed_primary_cam in _solved_ids:
            primary_cam_id = fixed_primary_cam
        else:
            primary_cam_id = max(
                cam_solutions, key=lambda cs: len(cs["solution"]["assignment"])
            )["cam_id"]
        anchor_solution = next(
            cs["solution"] for cs in cam_solutions if cs["cam_id"] == primary_cam_id
        )

        # A camera whose own independent search failed this frame is left out of
        # the fusion entirely — no Hungarian snap-recovery. Its confidence would
        # be zero anyway (no RANSAC/hypothesis validation behind a raw nearest-
        # neighbour snap), so it can't influence the fused pose either way; but an
        # unvalidated snap's pairs still get registered in claimed_blobs/
        # aux_assignments below, which can falsely block a *different* controller
        # from matching a blob it actually owns in this camera this frame. Zero
        # upside, real downside — so a failed camera is simply absent this frame.

        # Per-camera importance: each contributor's share of the fusion weight —
        # (1/sqrt(n_pairs)) * confidence, normalised to sum to 1 — i.e. how much
        # this frame's fused pose actually leaned on each camera. Replaces the old
        # "anchor camera" concept for reporting purposes.
        _raw_w = {
            e["camera"].camera_idx: float(e.get("confidence", 1.0)) / max(len(e["pairs"]), 1) ** 0.5
            for e in _fuse_in
        }
        _w_total = sum(_raw_w.values()) or 1.0
        _importance_str = "  ".join(
            f"cam{cid}={w / _w_total:.2f}" for cid, w in sorted(_raw_w.items())
        )

        logger.bind(cat="pose_fusion").debug(
            f"[{self.ctrl_name}] Joint fusion: {len(cam_solutions)} solved  "
            f"err={fused_err:.2f}px  lm={_t_fuse*1000:.1f}ms  "
            f"importance=[{_importance_str}]"
        )

        solution = dict(anchor_solution)
        solution["primary_cam"]  = primary_cam_id
        # Preserved for the self-cal feed below: the primary camera's own
        # independent PnP solve, untainted by any other (possibly still
        # uncalibrated) camera's contribution to the fused pose — see
        # _commit_fused_solution's self-cal block.
        solution["primary_T_world_ctrl"] = anchor_solution["T_world_ctrl"]
        solution["primary_error"]        = anchor_solution["error"]
        solution["T_world_ctrl"] = T_world_ctrl
        solution["error"]        = fused_err
        # Normalised per-camera fusion weight (sums to 1 across every solved
        # camera) — the same values behind the "importance=[...]" debug line
        # above, exposed here for consumers (e.g. visualization) that want to
        # reflect actual per-camera contribution rather than a primary/aux label.
        solution["camera_importance"] = {cid: w / _w_total for cid, w in _raw_w.items()}
        _other_assignments = {
            cs["cam_id"]: cs["solution"]["assignment"]
            for cs in cam_solutions if cs["cam_id"] != primary_cam_id
        }
        if _other_assignments:
            # Multiple cameras independently solved and were fused here -- each
            # OTHER camera's own primary assignment becomes this candidate's
            # aux evidence for that camera (blob-index-level, and strictly
            # richer than anything the anchor camera's own search separately
            # found via other_cameras_blobs).
            solution["aux_assignments"] = _other_assignments
            solution["aux_cameras"] = [(cid, len(pairs)) for cid, pairs in _other_assignments.items()]
        else:
            # Only the anchor camera contributed (e.g. update_cold_batch's
            # single-camera-winner selection, _select_best_cam_solution) --
            # fall back to the anchor's OWN brute-search aux-camera validation
            # (PoseSearcher.brute_search_tier's "6.7", scored against
            # other_cameras_blobs), the only real cross-camera evidence that
            # exists in this case. Explicitly defaulted (not just left as
            # whatever dict(anchor_solution) copied above) because anchor_
            # solution may not carry these keys at all (e.g. a synthetic/
            # non-brute solution dict) -- unconditionally overwriting with
            # {} / [] (the previous behaviour) silently discarded real
            # evidence when it *was* present, making every single-camera-
            # winner candidate report zero aux corroboration downstream
            # (_resolve_cold_conflicts' shared-blob/aux-projection checks and
            # _score/_fmt) regardless of what that camera's own search
            # actually found.
            solution["aux_assignments"] = anchor_solution.get("aux_assignments") or {}
            solution["aux_cameras"] = anchor_solution.get("aux_cameras") or []

        return solution

    def _propagate_pose_history(self, T_world_ctrl: Transform, frame_ts_ns: int) -> None:
        """Warm-start every camera's own prev_pose/pose_history/vel_ema from
        one world-frame pose -- the same computation regardless of where that
        pose came from: an accepted vision fusion (_commit_fused_solution) or
        pure IMU dead-reckoning while vision is lost (_mark_all_lost). Shared
        so the two "advance the search anchor" call sites can't silently
        diverge (this exact loop body used to live only in
        _commit_fused_solution -- see git history for the REVERTED note on why
        seeding from anything other than a stabilized pose is risky).

        Caller is responsible for any accept-specific bookkeeping on top of
        this (last_good_pose, prev_assignment, consecutive_failures) --
        _mark_all_lost's IMU-only case deliberately does none of that, since
        vision itself still didn't confirm anything this frame."""
        _beta = float(self._matching_cfg.get("pose_prediction_vel_ema_beta", 0.3))
        for _cid, _tracker in self.trackers.items():
            _T_cam_ctrl = self.cameras[_cid].T_world_cam.inverse().compose(T_world_ctrl)
            _rv_np, _ = cv2.Rodrigues(_T_cam_ctrl.R.astype(np.float32))
            _tv_np = _T_cam_ctrl.t.astype(np.float32)
            if _tracker.prev_pose is not None:
                _step = (_tv_np.reshape(3).astype(np.float64)
                         - np.asarray(_tracker.prev_pose[1], np.float64).reshape(3)).astype(np.float32)
                # vel_ema is a position-per-second RATE, not a raw per-call step —
                # pose_history is non-empty whenever prev_pose is set (both are
                # always written together, right here), so [0][2] is this
                # tracker's previous frame's real timestamp.
                _prev_ts_ns = int(_tracker.pose_history[0][2])
                _dt_s = (frame_ts_ns - _prev_ts_ns) / 1e9
                if _dt_s > 0:
                    _step_rate = _step / _dt_s
                    _tracker.vel_ema = (_beta * _step_rate + (1.0 - _beta) * _tracker.vel_ema
                                       if _tracker.vel_ema is not None else _step_rate)
                # else: degenerate/duplicate timestamp — leave vel_ema at its
                # previous value rather than dividing by a non-positive dt.
            _tracker.prev_prev_pose = _tracker.prev_pose
            _tracker.prev_pose      = (_rv_np.reshape(3, 1), _tv_np)
            _tracker.pose_history.appendleft((_rv_np.reshape(3, 1), _tv_np, frame_ts_ns))

    def _commit_fused_solution(
        self,
        solution: Dict,
        cam_solutions: List[dict],
        obs_src: Dict[int, np.ndarray],
        claimed_blobs: Optional[Dict[int, Set[int]]],
        frame_ts_ns: int,
        self_cal=None,
    ) -> None:
        """Side effects for a solution already chosen as the accepted result
        for this controller this frame: designated-primary bookkeeping,
        claimed-blob registration, self-cal feed, and propagation of the
        fused pose into every camera tracker's state. Call only once a
        solution has actually been accepted — for the plain single-candidate
        case (ControllerTracker.update(), TrackingSystem.update_warm_batch)
        that's unconditional (see _fuse_and_finalize below); for
        TrackingSystem.update_cold_batch's multi-candidate case, only after
        cross-controller conflict resolution has picked a winner.

        claimed_blobs=None skips claim registration entirely — used by the
        warm-warm and cold-cold batched paths, neither of which tracks
        cross-controller blob claims this way (warm-warm: each controller's
        own RANSAC/inlier scoring absorbs a shared blob; cold-cold: conflicts
        are resolved explicitly by TrackingSystem._resolve_cold_conflicts
        before this is ever called).

        Pose-fusion filter (see src/pose_fusion.py): when self._fusion_filter
        exists, the incoming solution is gated through PoseFusionFilter.try_update
        before any of the above runs. An ACCEPTED update proceeds exactly as
        without the filter, except every downstream consumer (state propagation,
        self-cal feed, and solution["T_world_ctrl"] itself, which callers read
        AFTER this returns for CSV/log/rerun output) sees the filter's corrected
        pose, not necessarily the raw vision solve. A REJECTED update skips claim
        registration and the self-cal feed entirely (an implausible candidate is
        not trustworthy ground truth for either), and state propagation instead
        uses the filter's own IMU-only prediction for this frame — this is the
        "smoothing term": the reported pose is now the filter's belief, never
        just whatever the raw vision solve happened to be. A persistently
        rejecting filter (should_force_cold_start) forces a full cold-start —
        see the escape-hatch block near the end of this method, plan Phase 5.
        """
        # NOTE: the IMU-only-coast budget (_mark_all_lost's elapsed-time check
        # against self._fusion_filter.last_update_ts_ns) is deliberately NOT
        # reset here -- last_update_ts_ns only advances inside the fusion
        # filter's own try_update, on a genuine accept (bootstrap/fail_open/
        # fused/cold-reacquire), never on implausible_reject/sibling_rejected
        # (see HeuristicPoseFusionFilter.try_update's own comment on this). _last_
        # imu_only_pose is likewise only cleared in the `if accepted:` block
        # below. This used to reset unconditionally right here, keyed off
        # "some camera found a geometric fit" rather than "the filter trusts
        # what it found" -- for HeuristicPoseFusionFilter in particular (no
        # hard gating -- see its own module docstring) a wildly implausible
        # candidate (identity swap, degenerate low-point fit) still reaches
        # this method every time vision hallucinates one, silently re-arming
        # a fresh coasting budget each time and preventing a genuinely lost
        # controller from ever being declared cold again (found investigating
        # a report of a controller "freezing" in place well past when
        # tracking should have gone lost).

        primary_cam_id     = solution["primary_cam"]
        anchor_assignment  = solution["assignment"]
        _other_assignments = solution["aux_assignments"]

        # Preserved before any fusion overwrite below, unconditionally (cheap --
        # same Transform object, no copy) -- solution["T_world_ctrl"] itself gets
        # overwritten in place with the filter's reported pose a few lines down,
        # with no other trace of the raw vision solve left in this dict afterward.
        # The rerun debug tool (visualize_pose_fusion) needs the raw pose to plot
        # alongside the fused one; every other consumer keeps reading
        # solution["T_world_ctrl"] exactly as before (found in review-adjacent
        # cross-check, not a behavior change).
        solution["vision_T_world_ctrl"] = solution["T_world_ctrl"]

        accepted = True
        if self._fusion_filter is not None:
            accepted = self._fusion_filter.try_update(solution, frame_ts_ns)
            # reported_R/reported_p can legitimately still be None here: a
            # brand new filter (self.R is None, never had any state at all)
            # whose very first candidate gets rejected before ever calling
            # _report() -- e.g. HeuristicPoseFusionFilter's bootstrap-branch
            # sibling-collision check -- has genuinely nothing to report yet,
            # unlike a WARM reject (implausible-jump, cold-reacquire-pending),
            # which always has a meaningful prior/IMU-predicted pose to fall
            # back on. T_world_ctrl stays None in that case rather than a
            # broken Transform(None, None) (crashed here and in every
            # downstream consumer before this fix, found on a real run) --
            # every consumer below and in main.py must treat T_world_ctrl is
            # None as "nothing to report this frame", same spirit as the
            # existing fusion_forced_cold_start hide-this-frame handling.
            _rep_R, _rep_p = self._fusion_filter.reported_R, self._fusion_filter.reported_p
            T_world_ctrl = Transform(_rep_R, _rep_p) if (_rep_R is not None and _rep_p is not None) else None
            solution["T_world_ctrl"] = T_world_ctrl  # downstream reporting reads this same dict
            if self._debug_pose_fusion:
                _dbg = self._fusion_filter.debug_snapshot(frame_ts_ns)
                solution["fusion_debug"] = _dbg
                _stride = int(self._debug_pose_fusion_cfg.get("path_stride", 4))
                solution["fusion_imu_path"] = self._fusion_filter.predict_dense(
                    frame_ts_ns, sample_every_n=_stride)
                # Console-visible trace for the SAME empirical-tuning purpose the rerun
                # tab exists for -- e.g. spotting a persistently-unconverged g_world
                # (predict()/predict_dense() silently fail-open/return None whenever
                # self._g_world_estimator.g_world is None, which makes every accept
                # "fail_open" and reported_p == the raw vision pose exactly, easy to
                # misread as a fusion bug rather than "IMU prediction isn't live yet").
                _pred_x = float(_dbg["pos_pred"][0]) if _dbg.get("pos_pred") is not None else None
                _vision_x = float(solution["vision_T_world_ctrl"].t[0])
                _fused_x = float(T_world_ctrl.t[0]) if T_world_ctrl is not None else None
                logger.bind(cat="pose_fusion").debug(
                    f"[{self.ctrl_name}] ts={frame_ts_ns} outcome={_dbg.get('outcome')} accepted={accepted} "
                    f"d2={_dbg.get('d2')} gate={_dbg.get('gate')} trust={_dbg.get('trust')} "
                    f"pred_x={_pred_x} vision_x={_vision_x} fused_x={_fused_x} "
                    f"g_world={'converged' if (self._g_world_estimator is not None and self._g_world_estimator.g_world is not None) else 'NOT converged'}"
                )
        else:
            T_world_ctrl = solution["T_world_ctrl"]
        # Downstream consumers (main.py's pose_csv/calibration_csv writers) that pair
        # T_world_ctrl with solution["error"]/["assignment"] need to know those fields
        # still describe the ORIGINAL vision candidate, not this frame's reported pose,
        # whenever a rejected update swapped T_world_ctrl for the filter's own IMU-only
        # prediction (found in review -- pairing a rejected pose with the discarded
        # candidate's error/assignment silently corrupts calibration-threshold data).
        solution["fusion_accepted"] = accepted

        # Reporting/self-cal anchor tracking (per-camera importance above is now the
        # reported signal; kept only for get_designated_primary_cameras()).
        self._designated_primary = primary_cam_id

        # ── Register claimed blobs (every camera that actually solved) ──────────
        if claimed_blobs is not None and accepted:
            for cs in cam_solutions:
                for b, _ in cs["solution"]["assignment"]:
                    claimed_blobs.setdefault(cs["cam_id"], set()).add(b)

        # ── Self-calibration feed ────────────────────────────────────────────────
        # Uses the primary camera's own independent PnP solve (primary_T_world_ctrl/
        # primary_error, set in _compute_fused_solution) as the 3D reference, NOT
        # the fused T_world_ctrl above — the fused pose is a confidence-weighted
        # blend across every solved camera, including the very aux cameras being
        # calibrated here. Feeding that blend back in as "ground truth" would be
        # circular: an aux camera's still-wrong intrinsics would leak into the
        # reference used to correct it. SelfCalibrator's own docstring assumes an
        # aux-camera-untainted primary solve; this is what actually provides one.
        if (self_cal is not None and accepted
                and primary_cam_id == self_cal.primary_camera.camera_idx):
            _primary_T_world_ctrl = solution["primary_T_world_ctrl"]
            _primary_err          = solution["primary_error"]
            _T_cam_ctrl_anchor = self.cameras[primary_cam_id].T_world_cam.inverse().compose(_primary_T_world_ctrl)
            _R_prim = _T_cam_ctrl_anchor.R.astype(np.float32)
            _rv_prim, _ = cv2.Rodrigues(_R_prim)
            _t_prim = _T_cam_ctrl_anchor.t.astype(np.float32)
            _sc_aux_obs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            for _aux_cid, _aux_pairs in _other_assignments.items():
                if len(_aux_pairs) < 3:
                    continue
                _led_ids   = np.array([lid for _, lid in _aux_pairs], dtype=np.int32)
                _blob_idxs = np.array([b   for b, _  in _aux_pairs], dtype=np.int32)
                _led_pos   = self.trackers[primary_cam_id].model.positions[_led_ids]
                _pts_prim  = (_R_prim @ _led_pos.T).T + _t_prim
                _blobs_aux = obs_src[_aux_cid][_blob_idxs]
                _sc_aux_obs[_aux_cid] = (_pts_prim, _blobs_aux)
            if _sc_aux_obs:
                self_cal.add_frame(
                    _rv_prim, _t_prim,
                    primary_error=_primary_err,
                    primary_inliers=len(anchor_assignment),
                    aux_observations=_sc_aux_obs,
                )

        # ── State propagation: every camera's own history is warm-started from
        # the one fused pose, using that camera's own assignment if it solved ──
        #
        # REVERTED (2026-09-04): tried seeding search-time pose_history from raw
        # vision on accepted updates instead of the fused T_world_ctrl, to
        # decouple search-neighbourhood centering from the fusion filter's own
        # trust/smoothing decisions (motivated by a real closed-loop problem --
        # see pose_fusion.py's try_update, low-reprojection-error floor comment).
        # An adversarial empirical audit (mocap-ground-truth-validated) found a
        # SEVERE regression this introduced on a different part of the same
        # recording (frame_range 2000-3000): the fused/smoothed pose had been
        # acting as a STABILIZER on the search neighbourhood -- removing it let
        # a drifting vision solve become self-reinforcing across ~40 frames
        # (left_controller: ~600ms, up to 840mm mistrack vs mocap; right_
        # controller: 34 dropped TRACKING LOST frames the fused-anchor version
        # tracked cleanly). Isolated conclusively to this change alone (bit-
        # identical vision output with the OTHER change of that session disabled
        # still reproduced the mistrack at full magnitude). Reverted rather than
        # shipped with a known severe failure mode; the underlying closed-loop
        # problem this was trying to fix is real but needs a more careful
        # solution (e.g. only anchor on vision when confidence is high, not
        # unconditionally on every accept) before revisiting.
        #
        # Skipped when T_world_ctrl is None (see its own construction above --
        # a brand new filter's very first candidate rejected before any state
        # ever existed, nothing to propagate from). prev_pose/pose_history
        # simply stay however they already were (None/empty for a genuine
        # first-ever reject), which is the correct "still no prior" state for
        # next frame's search -- crashed here otherwise (found on a real run).
        if T_world_ctrl is not None:
            self._propagate_pose_history(T_world_ctrl, frame_ts_ns)
        if accepted:
            # Vision produced a candidate the filter actually trusts -- this loss
            # streak (if any) is genuinely over; the next one gets its own fresh
            # IMU-only-propagation budget automatically, since _mark_all_lost's
            # elapsed-time check reads self._fusion_filter.last_update_ts_ns,
            # which the fusion filter itself only advances on a genuine accept
            # (bootstrap/fail_open/fused/cold-reacquire, never
            # implausible_reject/sibling_rejected -- see HeuristicPoseFusion-
            # Filter.try_update's own comment on why that distinction is load-
            # bearing here too, not just for consecutive_failures below). No
            # separate counter to reset any more.
            self._last_imu_only_pose = None
        # cam_id -> that camera's OWN solved dict (error/assignment), for
        # last_good_pose_quality below -- see that field's own comment.
        _cam_sol_by_cid = {cs["cam_id"]: cs["solution"] for cs in cam_solutions}
        for _cid, _tracker in self.trackers.items():
            # Assignment/failure bookkeeping reflects trust in the vision candidate
            # itself, not just "we have a pose to report" (that's the propagation
            # above, which always runs so next frame still gets a sane warm-start
            # prior). A rejected candidate is exactly the case where that trust
            # doesn't hold, so leave these at their last-accepted values rather
            # than seeding next frame's constrained search from an implausible
            # assignment. _mark_all_lost itself is still never called from
            # here (it also does vision-search bookkeeping that doesn't apply
            # when vision itself just succeeded) — see the persistent-reject
            # escape hatch right after this loop instead, plan Phase 5.
            if accepted:
                _tracker.last_good_pose = _tracker.prev_pose
                _tracker.last_good_pose_ts_ns = frame_ts_ns
                _own_asgn = (
                    anchor_assignment if _cid == primary_cam_id
                    else _other_assignments.get(_cid)
                )
                # Quality this camera's OWN solve reached when last_good_pose
                # was captured -- see last_good_pose_quality's own comment
                # (__init__) for why finalize_search's re-acquisition gate
                # needs this alongside the pose itself.
                _own_sol = _cam_sol_by_cid.get(_cid)
                if _own_asgn is not None and _own_sol is not None:
                    _tracker.last_good_pose_quality = (len(_own_asgn), float(_own_sol.get("error", 0.0)))
                if _own_asgn is not None:
                    _tracker.prev_assignment      = _own_asgn
                    _tracker.last_good_assignment = _own_asgn
                elif _tracker.prev_assignment is None:
                    _tracker.last_good_assignment = None
                _tracker.consecutive_failures = 0
                _tracker._consecutive_good_blob_frames = 0
                _tracker.tracking_lost_last_frame = False

        # ── Persistent-reject escape hatch (plan Phase 5) ───────────────────────
        # should_force_cold_start's two conditions (elapsed time past max_coast_s,
        # or too many consecutive rejects) were previously only ever checked from
        # _mark_all_lost, which only runs when vision finds NOTHING this frame
        # (cam_solutions empty). A controller vision keeps finding candidates for
        # every frame but this filter keeps rejecting never reaches that path at
        # all, so max_consecutive_rejects was unreachable in exactly the scenario
        # it exists for (found in review). Checked here instead, unconditionally
        # reachable on any reject regardless of why vision succeeded.
        #
        # Deliberately does MORE than reset the filter alone: the vision
        # candidates driving the rejections may themselves be anchored on a bad
        # camera-tracker prior (e.g. warm-tracking something plausible-looking
        # but wrong), which a filter-only reset wouldn't touch — a fresh
        # bootstrap would just accept the next frame's equally-anchored
        # candidate right back. So this also clears every camera tracker's own
        # prev_pose/pose_history, mirroring _mark_all_lost's grace-exhausted
        # branch, so next frame's ctrl_has_prior (main.py) sees a true
        # cold-start and triggers full blob-level re-detection + brute-force.
        #
        # consecutive_failures is ALSO pushed past tracking_lost_grace_frames
        # here (found in review) even though this frame's vision search itself
        # succeeded: finalize_search's re-acquisition plausibility gate only
        # applies its tight 0.5m/60deg check against last_good_pose while
        # consecutive_failures <= grace (see its own docstring — that check
        # assumes "last known good" was ~1 frame ago). Clearing prev_pose
        # above without also doing this would leave that invariant broken —
        # _reacquiring becomes True (prev_pose is None) but consecutive_failures
        # stays low, so the tight check would fire against a last_good_pose
        # that's actually up to max_coast_s/several-rejects stale, wrongly
        # rejecting the exact reacquisition candidate this escape hatch exists
        # to accept. Jumping straight past grace (rather than incrementing by
        # 1 like a normal per-frame failure) is correct, not a shortcut: by
        # the time should_force_cold_start fires, this controller has already
        # been in trouble for well over one grace period.
        if self._fusion_filter is not None and not accepted \
                and self._fusion_filter.should_force_cold_start(frame_ts_ns):
            solution["fusion_forced_cold_start"] = True  # debug-viz annotation; set BEFORE
                                                           # reset() below clears the state that
                                                           # actually triggered this
            # Read BEFORE reset() -- reset() clears self._fusion_filter._last to {}.
            # abs_reject is the narrow, high-precision identity-swap signature (see
            # pose_fusion.py's try_update): unlike an ordinary elapsed-time/
            # consecutive-rejects escape-hatch firing (an unremarkable prolonged
            # reacquisition failure, no reason to suspect the SIBLING controller),
            # an abs_reject firing means a real position was seen but for the WRONG
            # controller -- which by definition implicates the sibling's recent
            # trust too, not just this controller's. So ALSO cross-reset every
            # sibling's own filter/camera-tracker state (same reset shape as below)
            # rather than just this controller's -- see set_sibling_controllers's
            # docstring. This deliberately does NOT touch the sibling's `solution`
            # dict already computed this frame (no brittle same-frame retroactive
            # mutation) -- the sibling's own NEXT try_update() call naturally
            # re-evaluates against this now-clean state instead of stale memory.
            _is_abs_reject_swap = bool(self._fusion_filter._last.get("abs_reject"))
            _elapsed_s = ((frame_ts_ns - self._fusion_filter.last_update_ts_ns) / 1e9
                          if self._fusion_filter.last_update_ts_ns is not None else None)
            logger.bind(cat="matching_decisions").debug(
                f"[{self.ctrl_name}] PERSISTENT-REJECT ESCAPE HATCH fired: "
                f"elapsed_s={_elapsed_s} max_coast_s={self._fusion_filter._cfg.get('max_coast_s')} "
                f"consecutive_rejects={self._fusion_filter.consecutive_rejects} "
                f"max_consecutive_rejects={self._fusion_filter._cfg.get('max_consecutive_rejects')} "
                f"abs_reject_swap={_is_abs_reject_swap} "
                f"siblings_to_cross_reset={[s.ctrl_name for s in self._sibling_controllers] if _is_abs_reject_swap else []}"
            )
            self._fusion_filter.reset()
            _grace = int(self._matching_cfg.get('tracking_lost_grace_frames', 1))
            for _tracker in self.trackers.values():
                _tracker.clear_prior()
                _tracker.consecutive_failures = _grace + 1
                # clear_prior() deliberately leaves last_good_pose alone (it's meant
                # to survive an ORDINARY vision loss) -- but this escape hatch means
                # recent belief was PROVEN wrong (persistent implausible rejects, or
                # an outright identity-swap signature below), so last_good_pose was
                # almost certainly captured during that same contaminated stretch.
                # Same reasoning/fix as ControllerTracker.force_cold_start's own.
                _tracker.last_good_pose = None
                _tracker.last_good_pose_ts_ns = None
                _tracker.last_good_pose_quality = None
            if _is_abs_reject_swap:
                for _sibling in self._sibling_controllers:
                    if _sibling._fusion_filter is not None:
                        _sibling._fusion_filter.reset()
                    for _sib_tracker in _sibling.trackers.values():
                        _sib_tracker.clear_prior()
                        _sib_tracker.consecutive_failures = _grace + 1
                        _sib_tracker.last_good_pose = None
                        _sib_tracker.last_good_pose_ts_ns = None
                        _sib_tracker.last_good_pose_quality = None

        # Sampled once and shared below (found in review: the gravity-estimator feed
        # and _log_gravity_consistency were each independently interpolating the SAME
        # accel stream at this SAME frame_ts_ns).
        _accel_now = _interp_imu_sample(*self._accel_data, frame_ts_ns) if self._accel_data is not None else None

        # Feed the live gravity estimator (see LiveGravityEstimator's docstring:
        # accumulated incrementally on every ACCEPTED commit) — only meaningful
        # once a filter/gravity-estimator pair actually exists, and only from a
        # trustworthy pose.
        if self._fusion_filter is not None and accepted and self._g_world_estimator is not None \
                and self._gyro_data is not None and _accel_now is not None:
            _gyro_sample = _interp_imu_sample(*self._gyro_data, frame_ts_ns)
            if _gyro_sample is not None:
                self._g_world_estimator.observe(T_world_ctrl.R, _gyro_sample, _accel_now)

                # Second, ABSOLUTE-frame gravity accumulator for the headset-ego-motion-corrected
                # dead-reckoning path (see src.imu_data.predict_headset_relative_pose) -- COMMENTED
                # OUT, not deleted: direct total replacement by the empirically-validated
                # MOCAP_ROOM_G_WORLD constant (src/imu_data.py) -- the mocap room's Y axis was
                # confirmed gravity-aligned to within ~0.06 degrees across a real recording, so
                # this converges-over-time accumulator is unnecessary. Kept here for future
                # research if a future recording's room leveling ever needs re-estimating live.
                # if self._headset_mocap is not None and self._g_world_estimator_abs is not None:
                #     _T_wh = world_pose(self._headset_mocap, frame_ts_ns)
                #     if _T_wh is not None:
                #         self._g_world_estimator_abs.observe(_T_wh.R @ T_world_ctrl.R, _gyro_sample, _accel_now)

        # Stage 3 gravity-alignment diagnostic (log-only, see _log_gravity_consistency).
        # Skipped when T_world_ctrl is None (see its own construction above --
        # a brand new filter's very first candidate rejected before any state
        # ever existed) -- nothing to log/compare against; crashed here
        # otherwise (found on a real run, AttributeError on T_world_ctrl.R).
        if T_world_ctrl is not None:
            _log_gravity_consistency(
                self.ctrl_name, T_world_ctrl.R, frame_ts_ns,
                self._last_gravity_check_R, self._last_gravity_check_ts,
                self._accel_data, accel_now=_accel_now,
            )
            self._last_gravity_check_R  = T_world_ctrl.R
            self._last_gravity_check_ts = frame_ts_ns

    def _fuse_and_finalize(
        self,
        cam_solutions: List[dict],
        obs_src: Dict[int, np.ndarray],
        claimed_blobs: Optional[Dict[int, Set[int]]],
        frame_ts_ns: int,
        fixed_primary_cam: Optional[int] = None,
        self_cal=None,
    ) -> Dict:
        """Fuse then immediately commit — thin wrapper preserving the
        original combined behavior for the tier-round recovery path
        (update(), above) and the batched warm-warm path
        (TrackingSystem.update_warm_batch()). See _compute_fused_solution /
        _commit_fused_solution for the split and why it exists."""
        solution = self._compute_fused_solution(
            cam_solutions, obs_src, fixed_primary_cam=fixed_primary_cam,
        )
        self._commit_fused_solution(
            solution, cam_solutions, obs_src, claimed_blobs, frame_ts_ns, self_cal=self_cal,
        )
        return solution


def _inlier_discounted_error(error: float, total_pairs: int, min_inliers: float,
                              error_floor: float = 0.0) -> float:
    """A fit sitting right at the minimal-inlier floor has almost no spare degrees
    of freedom, so a near-zero residual there is not evidence of a correct match --
    it's just what an under-constrained fit looks like. Discount error linearly by
    how many multiples of the floor a candidate has: right at the floor -> no
    discount; 5x the floor -> error divided by 5. Shared by
    TrackingSystem._resolve_cold_conflicts (cross-controller candidates) and
    _select_best_cam_solution (one controller's own competing per-camera candidates).

    error_floor: sub-pixel reprojection-error differences are blob-centroid
    detection noise, not a real quality signal -- a 0.06px candidate is not
    meaningfully "better" than a 0.09px one, but the plain multiplicative
    discount above would still let that noise-level gap outrank a candidate
    with genuinely more corroborating evidence (more total_pairs). Clamping
    error to this floor before discounting means only a genuine, above-floor
    difference (one candidate is an actually worse fit, not just noisier)
    can move the score; two comparably-good fits are decided by total_pairs
    alone. 0.0 (default) preserves the original uncapped behavior -- callers
    pass a real floor (typically config's score_error_floor_px) explicitly."""
    return max(error, error_floor) * (min_inliers / max(total_pairs, 1))


def _select_best_cam_solution(cam_solutions: List[dict], min_inliers: float,
                               error_floor: float = 0.0) -> List[dict]:
    """When a controller's cameras each independently produced their own cold-start
    candidate this frame, pick exactly one winner by cross-camera-validated support
    (this camera's own primary inliers + aux_inliers -- already reprojected into
    every OTHER camera and matched against ITS raw blobs as part of this camera's
    own brute search; see PoseSearcher.brute_search_tier's "Aux-camera validation"
    step) and discard the rest, rather than joint-LM-blending every camera's own
    correspondences into one pose. A candidate built on a contaminated
    correspondence (e.g. two controllers' searches colliding on the same blob) will
    generically show weak aux support here, since its wrong 3D pose won't
    coincidentally explain another camera's real blob layout -- same failure mode
    TrackingSystem._resolve_cold_conflicts guards against across controllers,
    applied here across one controller's own cameras, before cross-controller
    resolution ever runs.

    error_floor: see _inlier_discounted_error.

    Returns cam_solutions unchanged if there's nothing to choose between.
    """
    if len(cam_solutions) <= 1:
        return cam_solutions

    def _score(cs: dict) -> float:
        sol = cs["solution"]
        total_pairs = len(sol.get("assignment") or []) + int(sol.get("aux_inliers") or 0)
        return _inlier_discounted_error(sol["error"], total_pairs, min_inliers, error_floor)

    return [min(cam_solutions, key=_score)]


# =========================================================
# 3. SYSTEM (multi-controller, multi-camera)
# =========================================================

class TrackingSystem:
    def __init__(self, controllers: List[ControllerModel], cameras: List[Camera],
                 matching_cfg: dict = None, geometry_cfg: dict = None,
                 geometry_cfg_per_ctrl: dict = None,
                 self_calibration_cfg: dict = None,
                 blob_detection_cfg: dict = None,
                 gyro_data: Optional[Dict[str, tuple]] = None,
                 accel_data: Optional[Dict[str, tuple]] = None,
                 lever_arm: Optional[Dict[str, np.ndarray]] = None,
                 g_world_estimator=None,
                 headset_mocap=None,
                 g_world_estimator_abs=None,
                 fusion_cfg: Optional[dict] = None,
                 debug_pose_fusion_cfg: Optional[dict] = None):

        self.cameras: Dict[int, Camera] = {cam.camera_idx: cam for cam in cameras}

        # Stage 3 IMU integration: {ctrl_name: (t_ns, values)}, already calibrated +
        # axis-corrected + clock-offset-corrected (see main.py) — or None/missing
        # entries when unavailable, in which case gyro/gravity-check logic no-ops.
        self._gyro_data:  Dict[str, tuple] = gyro_data or {}
        self._accel_data: Dict[str, tuple] = accel_data or {}
        # Pose-fusion filter wiring (see PoseFusionFilter / ControllerTracker) — one
        # shared LiveGravityEstimator across all controllers (gravity is a single
        # per-session unknown, not per-controller), one lever arm per controller.
        self._lever_arm:  Dict[str, np.ndarray] = lever_arm or {}
        self._g_world_estimator = g_world_estimator
        # Headset-ego-motion correction (see src.mocap_data.world_pose /
        # src.imu_data.predict_headset_relative_pose) -- both shared, non-per-controller objects,
        # threaded through to every ControllerTracker exactly like g_world_estimator above. None
        # unless mocap.enabled and a headset mocap trajectory actually loaded (main.py).
        self._headset_mocap = headset_mocap
        self._g_world_estimator_abs = g_world_estimator_abs
        self._fusion_cfg: dict = fusion_cfg or {}

        # Self-calibration: optionally apply saved extrinsics before tracker creation
        # so every tracker's T_world_cam starts with the correct (calibrated) value.
        self._self_cal: Optional[SelfCalibrator] = None
        self._self_cal = None
        sc_cfg = self_calibration_cfg or {}
        _sc_primary_idx: Optional[int] = None

        if sc_cfg.get("enabled", False):
            _sc_primary_idx = int(sc_cfg.get("primary_camera", 0))
            _aux_cam_idxs = sc_cfg.get("aux_cameras") or [
                cid for cid in self.cameras if cid != _sc_primary_idx
            ]
            _primary_cam = self.cameras.get(_sc_primary_idx)
            _aux_cams = [self.cameras[cid] for cid in _aux_cam_idxs if cid in self.cameras]

            if _primary_cam and _aux_cams:
                self._self_cal = SelfCalibrator(_primary_cam, _aux_cams, sc_cfg)

        self._matching_cfg: dict = matching_cfg or {}

        # Fixed primary camera exists only to serve self-calibration (it pins the
        # 3D reference every frame feeds SelfCalibrator.add_frame from — see
        # ControllerTracker._commit_fused_solution's self-cal block); with no
        # self-calibration running, there's no reason to override the normal
        # per-frame auto-select (most inlier pairs wins).
        self._fixed_primary_cam: Optional[int] = _sc_primary_idx

        self.trackers: Dict[Tuple[str, int], CameraTracker] = {}
        self.ctrl_trackers: Dict[str, ControllerTracker]    = {}

        _parallel_enabled = bool(self._matching_cfg.get('parallel_search_enabled', True))
        if _parallel_enabled:
            from src.parallel_search import register_pose_searcher_spec

        for ctrl in controllers:
            ctrl_cam_trackers: Dict[int, CameraTracker] = {}
            for cam in cameras:
                key = (ctrl.name, cam.camera_idx)
                geo = (geometry_cfg_per_ctrl or {}).get(ctrl.name) or geometry_cfg
                cam_tracker = CameraTracker(cam, ctrl, matching_cfg=matching_cfg, geometry_cfg=geo)
                self.trackers[key]           = cam_tracker
                ctrl_cam_trackers[cam.camera_idx] = cam_tracker
                if _parallel_enabled:
                    # Register the picklable construction spec (not the live
                    # PoseSearcher) — the pool uses 'spawn', so each worker builds
                    # its own PoseSearcher from these at startup. See
                    # src/parallel_search.py's module docstring for why not 'fork'.
                    register_pose_searcher_spec(key, cam, ctrl, geo, matching_cfg)
            self.ctrl_trackers[ctrl.name] = ControllerTracker(
                ctrl.name, self.cameras, ctrl_cam_trackers, matching_cfg=matching_cfg,
                gyro_data=self._gyro_data.get(ctrl.name),
                accel_data=self._accel_data.get(ctrl.name),
                lever_arm=self._lever_arm.get(ctrl.name),
                g_world_estimator=self._g_world_estimator,
                headset_mocap=self._headset_mocap,
                g_world_estimator_abs=self._g_world_estimator_abs,
                fusion_cfg=self._fusion_cfg,
                debug_pose_fusion_cfg=debug_pose_fusion_cfg,
            )

        # Wire each fusion filter to every OTHER enabled controller's own filter --
        # a second pass since every ControllerTracker (and its own PoseFusionFilter)
        # must already exist before any of them can reference the others. Used only
        # by PoseFusionFilter.try_update's bootstrap branch (see there / pose_fusion.py's
        # _looks_like_a_sibling): the per-controller-only gate has nothing to check a
        # fresh bootstrap candidate against, so this closes that gap by letting it check
        # a still-live SIBLING's own prediction instead — the cross-controller check the
        # plan's original per-controller-independent design didn't have.
        for ctrl in controllers:
            tracker = self.ctrl_trackers[ctrl.name]
            if tracker._fusion_filter is not None:
                siblings = [
                    other._fusion_filter for name, other in self.ctrl_trackers.items()
                    if name != ctrl.name and other._fusion_filter is not None
                ]
                tracker._fusion_filter.set_siblings(siblings)
            ctrl_siblings = [
                other for name, other in self.ctrl_trackers.items() if name != ctrl.name
            ]
            tracker.set_sibling_controllers(ctrl_siblings)

        # Independent of parallel_search_enabled — lets blob-detection parallelism be
        # turned off on its own (e.g. if IPC/pickling the debug canvases turns out to
        # dominate the warm path's actual per-LED-ROI compute) without giving up
        # pose-search parallelism, which still needs the pool regardless.
        self._blob_parallel_enabled = (
            _parallel_enabled
            and bool(self._matching_cfg.get('parallel_blob_detection_enabled', True))
            and blob_detection_cfg is not None
        )
        if self._blob_parallel_enabled:
            from src.parallel_search import register_blob_detector_spec
            for cam in cameras:
                register_blob_detector_spec(cam.camera_idx, blob_detection_cfg,
                                             cam.width, cam.height)

        # ── Persistent process pool for parallel cheap-search / brute-tier dispatch,
        # and (when enabled) parallel blob detection ──────────────────────────────
        self._pool = None
        if _parallel_enabled:
            from src.parallel_search import create_pool, warmup_pool
            from src import debug_config
            _workers_cfg = self._matching_cfg.get('parallel_search_workers')
            _n_workers = int(_workers_cfg) if _workers_cfg is not None else max(1, len(self.cameras))
            self._pool = create_pool(_n_workers, debug_cfg=debug_config.get_config())
            import atexit
            atexit.register(self._pool.shutdown)
            # Force every worker to spawn now (interpreter boot + heavy imports +
            # PoseSearcher/BlobDetector construction) instead of lazily on the first
            # tracked frame — spawn only starts a worker when it has a task to run.
            warmup_pool(self._pool, _n_workers)

    def get_pool(self):
        """The persistent process pool (or None if parallel search is disabled).
        Callers submitting blob-detection tasks must also check
        blob_parallel_enabled — the pool can exist for pose search alone while blob
        parallelism is independently turned off, in which case workers never built a
        _BLOB_DETECTORS registry."""
        return self._pool

    @property
    def blob_parallel_enabled(self) -> bool:
        return self._blob_parallel_enabled

    def shutdown(self) -> None:
        """Cleanly shut down the process pool, if one was created. Safe to call more
        than once; also registered via atexit as a safety net."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def get_designated_primary_cameras(self) -> Dict[str, Optional[int]]:
        """Return {ctrl_name: anchor_cam_id} as of the last successful tracking frame.

        Informational only — every camera searches independently every frame regardless
        of this value. None means no successful frame has been tracked yet.
        """
        return {name: ct._designated_primary for name, ct in self.ctrl_trackers.items()}

    def get_predicted_led_projections_per_camera(
        self,
        frame_ts_ns: int,
    ) -> Tuple[Dict[int, Dict[str, Optional[np.ndarray]]], Dict[int, Dict[str, float]],
               Dict[int, Dict[str, float]], Dict[int, Dict[str, bool]]]:
        """Return (proj_hints, vel_hints, radius_hints, search_eligible).

        frame_ts_ns: the current frame's real capture timestamp (nanoseconds,
        parsed from the frame's filename) — predictions are extrapolated to this
        exact elapsed time, not an assumed uniform frame step.

        proj_hints: {cam_id: {ctrl_name: Nx5 array or None}}
          Each row: [proj_x, proj_y, depth_m, facing_cos, led_id]
          for each LED visible from the predicted pose.
          None when no prior pose exists for this (ctrl, cam) pair, i.e. zero
          geometrically visible LEDs. A camera that's visible but lost the
          warm-cam-cap ranking below still has a real (non-None) entry here —
          see search_eligible.

        vel_hints, radius_hints: {cam_id: {ctrl_name: v_px / search_radius_px}}
          Estimated pixel displacement / effective search radius, for whatever
          camera main.py actually searches. 0.0 / base radius when no
          prediction is available. Retained (not reset) for cap-losing
          cameras too, since _build_blackout_images uses radius_hints to size
          the region it blacks out for them — see search_eligible.

        search_eligible: {cam_id: {ctrl_name: bool}}
          Whether main.py should actually run blob detection / pose search on
          this (cam, ctrl) pair this frame. False whenever proj_hints is None
          (nothing to search) OR the warm-cam-cap ranking demoted this camera
          in favor of better candidates — in the latter case proj_hints/
          vel_hints/radius_hints are still the real, non-reset values, kept
          specifically so a cold-starting OTHER controller's
          _build_blackout_images can still mask this controller's expected
          LED neighborhoods there, even though this controller itself won't
          search that camera this frame (warm-cold cross-controller
          contamination guard; irrelevant in the warm-warm case since
          _build_blackout_images is only invoked for a controller with no
          prior at all — see its call site in main.py).
        """
        ctrl_names   = sorted({ctrl for ctrl, _ in self.trackers})
        _facing_deg  = float(self._matching_cfg.get('led_facing_angle_deg', 86.0))
        # Any geometric visibility at all (>= 1 LED) is candidacy, not a search
        # guarantee — a controller can end up warm-tracked in 3-4 cameras when 2
        # would do. Rather than pre-filtering candidacy on a per-camera-independent
        # LED-count estimate (which has been observed to wrongly exclude a camera
        # whose real image would actually be the best one to search — see the cap
        # comment below), every camera with >=1 visible LED is a ranking candidate:
        # cameras viewing the controller too edge-on are excluded outright
        # (warm_cam_cap_min_facing_deg), survivors ranked by mean LED distance,
        # top max_warm_search_cameras kept.
        _max_warm_cams = int(self._matching_cfg.get('max_warm_search_cameras', 0)) or None
        _min_facing_cos_gate = float(np.cos(np.radians(
            float(self._matching_cfg.get('warm_cam_cap_min_facing_deg', 80.0)))))
        result:          Dict[int, Dict[str, Optional[np.ndarray]]] = {}
        vel_hints:       Dict[int, Dict[str, float]]                = {}
        radius_hints:    Dict[int, Dict[str, float]]                = {}
        search_eligible: Dict[int, Dict[str, bool]]                 = {}
        # (ctrl_name, cam_id) -> mean perpendicular distance (m) from the
        # controller's visible LEDs to that camera's optical axis (NOT Euclidean
        # range to the camera center — see the lateral_dist comment below) — used
        # below (together with _center_facing) to rank cameras for the warm-search
        # cap.
        _center_dist:   Dict[Tuple[str, int], float]              = {}
        # (ctrl_name, cam_id) -> mean facing_cos over the controller's visible
        # LEDs (how squarely they face this camera, averaged — NOT summed, so
        # LED count doesn't leak into what's meant to be a pure angle score) —
        # used below purely as a gate (grazing cameras excluded outright), not
        # as the ranking key: a camera can score well here on a couple of
        # borderline LEDs while still lacking enough of them to ever actually
        # match — see the cap comment below.
        _center_facing: Dict[Tuple[str, int], float]              = {}

        # Pre-read all four expansion terms once — same values used in track()
        _base_r = float(self._matching_cfg.get('proximity_expansion_px', 8.0))
        _vel_k  = float(self._matching_cfg.get('proximity_expansion_velocity_k', 0.0))
        _unc_k  = float(self._matching_cfg.get('proximity_expansion_uncertainty_k', 0.0))
        _dpt_k  = float(self._matching_cfg.get('proximity_expansion_depth_k', 0.0))
        # Same weight_decay cheap_search_core passes to _predict_pose — keep this
        # projection hint aligned with what proximity_search actually matches
        # against, or the search ROI drawn here silently disagrees with cheap_search_core's
        # own (vel_ema-corrected) prediction under fast/oscillating motion.
        _weight_decay = float(self._matching_cfg.get('pose_prediction_weight_decay', 0.7))

        for cam_id, camera in self.cameras.items():
            proj_per_ctrl:   Dict[str, Optional[np.ndarray]] = {}
            vel_per_ctrl:    Dict[str, float]                = {}
            radius_per_ctrl: Dict[str, float]               = {}
            for ctrl_name in ctrl_names:
                tracker = self.trackers.get((ctrl_name, cam_id))
                pred = (CameraTracker._predict_pose(
                            tracker.pose_history,
                            frame_ts_ns,
                            weight_decay=_weight_decay,
                            vel_ema_rate=tracker.vel_ema,
                        ) if tracker else None)
                if pred is None:
                    logger.bind(cat="matching_decisions").debug(
                        f"[{ctrl_name} | cam {cam_id}] search_eligible=False: no pose_history "
                        f"(pose_history len={len(tracker.pose_history) if tracker else 'no-tracker'})"
                    )
                    proj_per_ctrl[ctrl_name]   = None
                    vel_per_ctrl[ctrl_name]    = 0.0
                    radius_per_ctrl[ctrl_name] = _base_r
                    continue
                rvec_pred, tvec_pred = pred
                _ph = tracker.pose_history
                # dt from the last known pose to THIS frame -- used below ONLY
                # for the debug log's dt_target field. tvec_pred (the predicted
                # CENTER, from _predict_pose above) already correctly folds this
                # in via vel_ema_rate * dt_target; it must NOT also be baked
                # into the margin/radius term below (vel_rate_3d) -- see that
                # computation's own comment for why (same bug/fix as
                # cheap_search_core's _v_px, src/controller.py).
                _dt_target = max((frame_ts_ns - int(_ph[0][2])) / 1e9, 0.0) if _ph else 0.0
                # Speed estimate (m/s) for the margin term below -- deliberately
                # NOT scaled by _dt_target: doing that used to make the margin
                # swing ~2x with this recording's own 30/60fps cadence
                # oscillation for identical real hand speed, since dt_target
                # itself swings ~2x frame to frame (found investigating a
                # reported oscillating search-neighbourhood size). The
                # predicted CENTER above already correctly incorporates
                # dt_target; this is a separate, deliberately dt-INDEPENDENT
                # "how fast is it moving right now" term.
                if tracker.vel_ema is not None:
                    vel_rate_3d = tracker.vel_ema.astype(np.float64)
                elif len(_ph) >= 2:
                    _dt_hist = (int(_ph[0][2]) - int(_ph[1][2])) / 1e9
                    _step    = (np.asarray(_ph[0][1], np.float64).reshape(3)
                                - np.asarray(_ph[1][1], np.float64).reshape(3))
                    vel_rate_3d = (_step / _dt_hist) if _dt_hist > 0 else np.zeros(3, np.float64)
                else:
                    vel_rate_3d = np.zeros(3, np.float64)

                R_pred, _ = cv2.Rodrigues(rvec_pred.reshape(3, 1))
                R_pred    = R_pred.astype(np.float32)
                tvec_pred = tvec_pred.reshape(3).astype(np.float32)

                positions = tracker.model.positions
                normals   = tracker.model.normals

                vis_ids = np.where(_visible_mask(
                    R_pred, tvec_pred, positions, normals, tracker._geometry,
                    cam_K=camera.camera_matrix, cam_dc=camera.dist_coeffs,
                    cam_w=camera.width, cam_h=camera.height, cam_rpmax=camera.rpmax,
                    cam_is_fisheye=camera.is_fisheye,
                    facing_threshold_deg=_facing_deg,
                ) >= 1.0)[0]
                if len(vis_ids) == 0:
                    # Nothing to project or rank. The LED-count floor that used to
                    # live here has moved entirely into the warm-cam-cap ranking
                    # pass below, which now treats every camera with >=1 visible
                    # LED as a candidate (see comment above).
                    logger.bind(cat="matching_decisions").debug(
                        f"[{ctrl_name} | cam {cam_id}] search_eligible=False: predicted pose "
                        f"tvec={tvec_pred.tolist()} has 0 geometrically-visible LEDs "
                        f"(out of frustum / facing away / behind camera) "
                        f"pose_history_ts0={int(_ph[0][2]) if _ph else None} dt_target={_dt_target:.4f}s"
                    )
                    proj_per_ctrl[ctrl_name]   = None
                    vel_per_ctrl[ctrl_name]    = 0.0
                    radius_per_ctrl[ctrl_name] = _base_r
                    continue

                proj_pts = _project_points(
                    rvec_pred, tvec_pred, positions[vis_ids],
                    camera.camera_matrix, camera.dist_coeffs,
                    is_fisheye=camera.is_fisheye,
                )  # (M, 2)

                led_cam     = (R_pred @ positions[vis_ids].T).T + tvec_pred  # (M, 3)
                depths      = led_cam[:, 2]                                   # (M,)
                led_ranges  = np.linalg.norm(led_cam, axis=1)                 # (M,) true 3D distance to camera center
                view_dir    = led_cam / (led_ranges.reshape(-1, 1) + 1e-8)
                normals_cam = (R_pred @ normals[vis_ids].T).T                 # (M, 3)
                facing_cos  = -(normals_cam * view_dir).sum(axis=1)           # positive = faces cam
                # Perpendicular distance from each LED to this camera's optical axis
                # (the line through its center along local +z): for a point (x,y,z)
                # in camera frame that's sqrt(x^2+y^2), i.e. range with the
                # along-axis (depth) component removed. depths > 1e-6 is already
                # guaranteed per-LED by _visible_mask, so "behind the projection
                # surface" is already excluded before we get here — no separate
                # reject step needed.
                lateral_dist = np.linalg.norm(led_cam[:, :2], axis=1)         # (M,)

                proj_per_ctrl[ctrl_name] = np.column_stack([
                    proj_pts.astype(np.float32),
                    depths.astype(np.float32).reshape(-1, 1),
                    facing_cos.astype(np.float32).reshape(-1, 1),
                    vis_ids.astype(np.float32).reshape(-1, 1),
                ])  # (M, 5): proj_x, proj_y, depth_m, facing_cos, led_id

                # Mean perpendicular (off-axis) distance from the visible LEDs to
                # this camera's optical axis — used as the ranking key below,
                # replacing mean Euclidean range. A camera viewing the controller
                # near edge-on has almost all its range in the lateral component
                # (depth ~0), so this collapses toward the full range for such a
                # camera, while a camera viewing it near head-on has most of its
                # range in depth, so this collapses toward ~0 — a much starker
                # separation than raw Euclidean distance gave between e.g. cam0/1
                # (near head-on) and cam2/3 (near edge-on, ~180°-rotated) in
                # practice. Mean facing_cos is kept as the gate below, not used
                # for ranking. Both are means, not sums, so LED count doesn't leak
                # into either score.
                _center_dist[(ctrl_name, cam_id)]   = float(np.mean(lateral_dist))
                _center_facing[(ctrl_name, cam_id)] = float(np.mean(facing_cos))
                _depth_pred = max(float(tvec_pred[2]), 0.1)
                # Same fixed reference period as cheap_search_core's own _v_px --
                # keep the two in lockstep (see its comment for why a config-
                # driven default of proximity_expansion_velocity_ref_dt_s, not
                # a scale re-derived here independently).
                _ref_dt     = float(self._matching_cfg.get('proximity_expansion_velocity_ref_dt_s', 0.0333))
                _v_px       = float(np.linalg.norm(vel_rate_3d)) * _ref_dt * camera.fx / _depth_pred
                vel_per_ctrl[ctrl_name] = _v_px

                # Full effective blob-detection search radius — mirrors track() expansion logic
                _r = _base_r + _vel_k * _v_px
                if _unc_k > 0.0:
                    _r += _unc_k / max(len(tracker.pose_history), 1)
                if _dpt_k > 0.0:
                    _r += _dpt_k / _depth_pred
                radius_per_ctrl[ctrl_name] = _r

            result[cam_id]          = proj_per_ctrl
            vel_hints[cam_id]       = vel_per_ctrl
            radius_hints[cam_id]    = radius_per_ctrl
            search_eligible[cam_id] = {
                ctrl_name: proj_per_ctrl[ctrl_name] is not None for ctrl_name in ctrl_names
            }

        # ── Cap warm-search cameras per controller ──────────────────────────────
        # Two-stage selection among cameras theoretically visible for this
        # controller: (1) GATE — drop any camera whose mean facing_cos falls
        # below warm_cam_cap_min_facing_deg's cosine outright, regardless of
        # distance (a grazing view gives worse blobs — dimmer, more elongated,
        # noisier centroid — no matter how close the camera is); (2) RANK the
        # survivors by mean perpendicular (off-axis) distance from the visible
        # LEDs to each camera's optical axis, closest wins, keep the top
        # max_warm_search_cameras (see lateral_dist comment above). Three
        # single-metric schemes were tried and rejected first: an original
        # visible-LED-count / facing_cos-sum score let LED count leak into the
        # ranking; pure mean-facing_cos ranking let a camera with a great angle
        # on just a couple of borderline LEDs outrank one with plenty of LEDs
        # at a merely-good angle; mean Euclidean range to the camera center
        # worked much better but still occasionally lost on razor-thin margins
        # (e.g. 0.280m vs 0.286m) to a camera that structurally never produces
        # a match — a ~180°-rotated camera's near-edge-on view keeps its
        # Euclidean range deceptively similar to a well-facing camera's, since
        # range doesn't distinguish "close and off to the side" from "close and
        # dead ahead". Off-axis distance separates these sharply (observed
        # 0.048m/0.188m for two good cameras vs 0.278m for the bad one, on the
        # exact frame the Euclidean version mis-ranked) because it collapses
        # toward the full range for an edge-on view and toward ~0 for a
        # head-on one. Gating on facing_cos rather than folding it into a
        # blended score keeps each metric doing the job it's actually good at.
        # If every candidate fails the gate (all views are marginal), the
        # gate is skipped rather than demoting all of them, so far-fetched
        # geometry never leaves a controller with zero search cameras.
        # Demoted cameras are marked search_eligible=False — NOT reset to None —
        # so main.py's Phase 1 skips real blob detection/search there (via
        # search_eligible, not proj_hints), exactly like the "too few visible
        # LEDs" branch above, EXCEPT proj_hints/vel_hints/radius_hints are left
        # as the real, computed values rather than nulled out. That's
        # deliberate, not an oversight: this controller's own tracker won't
        # search this camera this frame, but if some OTHER controller has no
        # prior at all this frame and is about to run a full-image cold
        # brute-force search on this same camera, _build_blackout_images (in
        # main.py) needs this controller's real predicted LED positions/radii
        # here to black them out first — otherwise a demoted-but-still-real
        # bright blob sits fully exposed for the cold controller's search to
        # accidentally match as one of its own LEDs. That blackout is scoped
        # tightly (just the predicted-LED neighborhoods, not the whole frame)
        # and only matters for one frame at 30fps, so the cold controller
        # still has the rest of the image plus every subsequent frame to
        # reacquire — an acceptable trade against a false cross-controller
        # match, which is not recoverable the same way. _build_blackout_images
        # itself needs no changes for this: it already keys purely off
        # proj_hints being non-None, and only runs at all for a controller
        # with no prior anywhere this frame (the warm-cold case specifically —
        # irrelevant, and never invoked, in the warm-warm case). A cold
        # controller has no prediction on any camera, so none of this affects
        # its own brute-force reacquisition (never gated by proj_hints or
        # search_eligible at all).
        #
        # Candidacy here (membership in _all_cids, i.e. result[cid][ctrl_name] is
        # not None) is gated ONLY on >=1 geometrically visible LED — a camera
        # predicting too few LEDs to ever reach min_inliers downstream is no
        # longer pre-excluded before this ranking runs. If such a camera still
        # wins the cap (closest, passes the facing gate), it's kept as a real
        # search candidate: main.py runs real blob detection and a real
        # proximity/brute-force search on it, which is mathematically guaranteed
        # to return None via pose_search.py's own _c_prox_min_inliers /
        # _c_brute_min_inliers checks if the LED count is genuinely too low — so
        # it can never inject a bad pose into fusion, and it contributes zero
        # cam_solutions weight there, exactly like the "no candidate" case. What
        # it gains over being pre-excluded: real detection, a real (if losing)
        # search, and real visualization output for a camera whose actual image
        # often turns out better than this geometric estimate predicts.
        if _max_warm_cams is not None:
            for ctrl_name in ctrl_names:
                _all_cids = [cid for cid in result if result[cid][ctrl_name] is not None]
                if _all_cids:
                    logger.bind(cat="matching_decisions").debug(
                        f"[{ctrl_name}] warm-cam-cap candidates: " + "  ".join(
                            f"cam{cid}(n={len(result[cid][ctrl_name])},"
                            f"dist={_center_dist[(ctrl_name, cid)]:.3f}m,"
                            f"facing={_center_facing[(ctrl_name, cid)]:.3f})"
                            for cid in sorted(_all_cids, key=lambda c: _center_dist[(ctrl_name, c)])
                        )
                    )
                if len(_all_cids) <= _max_warm_cams:
                    continue
                _gated_cids = [
                    cid for cid in _all_cids
                    if _center_facing[(ctrl_name, cid)] >= _min_facing_cos_gate
                ] or _all_cids  # safety net: don't starve the controller if every view is marginal
                _kept_cids = {
                    cid for _, cid in sorted((_center_dist[(ctrl_name, cid)], cid)
                                              for cid in _gated_cids)[:_max_warm_cams]
                }
                for cid in _all_cids:
                    if cid in _kept_cids:
                        continue
                    logger.bind(cat="matching_decisions").debug(f"[{ctrl_name}] warm-cam-cap demoting cam{cid}")
                    search_eligible[cid][ctrl_name] = False

        return result, vel_hints, radius_hints, search_eligible

    def solved_led_geometry(
        self,
        ctrl_name: str,
        cam_id: int,
        T_world_ctrl: Transform,
        led_ids,
    ) -> Dict[int, Tuple[float, float]]:
        """Depth (m) and facing_cos for specific LEDs, computed from an
        already-solved T_world_ctrl rather than the pre-match extrapolated
        prediction (proj_hints from get_predicted_led_projections_per_camera)
        — that prediction is the prior fed *into* pose search, and drifts from
        the truth exactly when the controller is rotating/accelerating fast,
        which is the regime calibration data most needs to be accurate for.
        """
        tracker = self.trackers.get((ctrl_name, cam_id))
        if tracker is None:
            return {}
        camera     = self.cameras[cam_id]
        T_cam_ctrl = camera.T_world_cam.inverse().compose(T_world_ctrl)
        R_c = T_cam_ctrl.R.astype(np.float32)
        t_c = T_cam_ctrl.t.reshape(3).astype(np.float32)

        ids       = np.asarray(list(led_ids), dtype=int)
        positions = tracker.model.positions[ids]
        normals   = tracker.model.normals[ids]

        led_cam     = (R_c @ positions.T).T + t_c
        depths      = led_cam[:, 2]
        view_dir    = led_cam / (np.linalg.norm(led_cam, axis=1, keepdims=True) + 1e-8)
        normals_cam = (R_c @ normals.T).T
        facing_cos  = -(normals_cam * view_dir).sum(axis=1)

        return {int(lid): (float(d), float(fc)) for lid, d, fc in zip(ids, depths, facing_cos)}

    def get_ctrl_processing_order(self) -> List[str]:
        """Return controller names in the same priority order used by update()."""
        ctrl_names = sorted({ctrl for ctrl, _ in self.trackers})

        def _has_prior(name: str) -> bool:
            return any(
                t.prev_pose is not None
                for (cname, _), t in self.trackers.items()
                if cname == name
            )

        def _order_key(name: str):
            return (0 if _has_prior(name) else 1, name)

        return sorted(ctrl_names, key=_order_key)

    def update(
        self,
        observations_per_camera: Dict[int, np.ndarray],
        frame_ts_ns: int,
        radii_per_camera: Optional[Dict[int, np.ndarray]] = None,
        brightnesses_per_camera: Optional[Dict[int, np.ndarray]] = None,
        per_ctrl_observations: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        per_ctrl_radii: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        per_ctrl_brightnesses: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        ctrl_name_filter: Optional[str] = None,
        allow_brute: bool = True,
        force_brute: bool = False,
    ) -> Dict[str, Optional[Dict]]:
        """
        Run tracking for every controller using a single primary camera.

        frame_ts_ns: the current frame's real capture timestamp (nanoseconds,
        parsed from the frame's filename) — required; used to extrapolate pose
        history against actual elapsed time rather than an assumed uniform step.

        Blob ownership model: observations_per_camera is never mutated. Instead,
        claimed_blobs tracks which original blob indices have been consumed by
        earlier controllers. Each controller receives a pre-filtered view of
        unclaimed blobs; assignments are remapped to original indices by ControllerTracker.

        Controllers with a prior pose run first so their high-confidence claims
        are registered before cold-starting controllers search.

        Returns {ctrl_name: solution_or_None}. The solution dict gains a
        "primary_cam" key with the index of the camera that produced the pose.
        """
        results: Dict[str, Optional[Dict]] = {}

        ordered = self.get_ctrl_processing_order()
        if ctrl_name_filter is not None:
            ordered = [n for n in ordered if n == ctrl_name_filter]

        # Blob ownership: original observation indices claimed by earlier controllers.
        claimed_blobs: Dict[int, Set[int]] = {}

        def _filter_cam(cam_id: int, obs_map: dict, rad_map: dict, brt_map: dict):
            """Return (filtered_blobs, filtered_radii, filtered_brts, orig_indices)
            excluding claimed blobs. Returns (None,None,None,[]) if empty."""
            obs = obs_map.get(cam_id)
            if obs is None or len(obs) == 0:
                return None, None, None, []
            excluded = claimed_blobs.get(cam_id, set())
            keep = [i for i in range(len(obs)) if i not in excluded]
            if not keep:
                return None, None, None, []
            k = np.array(keep, dtype=np.int32)
            f_blobs = obs[k]
            _r = rad_map.get(cam_id) if rad_map else None
            f_radii = _r[k] if _r is not None else None
            _b = brt_map.get(cam_id) if brt_map else None
            f_brts  = _b[k] if _b is not None else None
            return f_blobs, f_radii, f_brts, keep

        _cfg = self._matching_cfg

        # ── Per-controller tracking ────────────────────────────────────────────
        for ctrl_name in ordered:
            if per_ctrl_observations is not None and ctrl_name in per_ctrl_observations:
                _obs_src = per_ctrl_observations[ctrl_name]
                _rad_src = (per_ctrl_radii or {}).get(ctrl_name) or {}
                _brt_src = (per_ctrl_brightnesses or {}).get(ctrl_name) or {}
            else:
                _obs_src = observations_per_camera
                _rad_src = radii_per_camera or {}
                _brt_src = brightnesses_per_camera or {}

            avail: Dict[int, tuple] = {}
            for cam_id in self.ctrl_trackers[ctrl_name].trackers:
                fb, fr, fbt, keep = _filter_cam(cam_id, _obs_src, _rad_src, _brt_src)
                if fb is not None:
                    avail[cam_id] = (fb, fr, fbt, keep)

            # Build cross-controller occluder dict from controllers matched this frame.
            _occluders_per_cam = None
            if bool(_cfg.get('cross_controller_occlusion', False)) and len(ordered) > 1:
                for _occ_ctrl in ordered:
                    if _occ_ctrl == ctrl_name:
                        break
                    _occ_result = results.get(_occ_ctrl)
                    if not (_occ_result and _occ_result.get('T_world_ctrl') is not None):
                        continue
                    _occ_T_world = _occ_result['T_world_ctrl']
                    _occ_tracker = next(
                        (t for (cn, _), t in self.trackers.items() if cn == _occ_ctrl), None,
                    )
                    if _occ_tracker is None:
                        continue
                    _occluders_per_cam = {}
                    for _cam_id, _cam in self.cameras.items():
                        if _cam.T_world_cam is None:
                            continue
                        _T_cam_occ = _cam.T_world_cam.inverse().compose(_occ_T_world)
                        _occluders_per_cam[_cam_id] = (
                            _T_cam_occ.R.astype(np.float32),
                            _T_cam_occ.t.astype(np.float32),
                            _occ_tracker._geometry,
                        )
                    break

            results[ctrl_name] = self.ctrl_trackers[ctrl_name].update(
                avail, _obs_src, _rad_src, _brt_src,
                claimed_blobs,
                frame_ts_ns,
                fixed_primary_cam=self._fixed_primary_cam,
                occluders_per_cam=_occluders_per_cam,
                self_cal=self._self_cal,
                pool=self._pool,
                allow_brute=allow_brute,
                force_brute=force_brute,
            )

        return results

    def _build_extrapolated_occluders(
        self, ctrl_names: List[str], frame_ts_ns: int,
    ) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray, object]]]:
        """Return {ctrl_name: {cam_id: (R_occ_in_cam, t_occ_in_cam, geometry)}}
        — one controller's occlusion-relevant pose per camera, built from its
        own *extrapolated* (predicted) pose rather than a same-frame
        confirmed solution. This is what makes cross-controller occlusion
        usable from update_warm_batch: unlike ControllerTracker.update()'s
        sequential occluder-building (controller.py, in update()), which
        needs an earlier-processed controller's already-solved T_world_ctrl
        for THIS frame, extrapolation needs nothing from this frame at all —
        every controller's dict can be built independently, before any
        cheap-search task even runs, with no ordering dependency between
        controllers. Cost is a handful of cheap host-side numpy/Rodrigues
        ops (one _predict_pose + up to len(self.cameras) Transform
        compositions per controller) — no image or blob data, no pool
        round-trip.

        Gated on the existing cross_controller_occlusion config flag (same
        one the sequential path uses). Returns {} when disabled, when fewer
        than 2 controllers are given, or when a controller has no pose
        history yet to extrapolate from (its own entry is simply absent —
        callers already treat occluders_per_cam.get(...) returning None as
        "no occluder this frame", the same as the sequential path does for a
        controller that hasn't solved yet).
        """
        if not bool(self._matching_cfg.get('cross_controller_occlusion', False)) or len(ctrl_names) < 2:
            return {}

        _weight_decay = float(self._matching_cfg.get('pose_prediction_weight_decay', 0.7))
        occluders_by_ctrl: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray, object]]] = {}
        for ctrl_name in ctrl_names:
            _any_tracker = next(iter(self.ctrl_trackers[ctrl_name].trackers.values()), None)
            if _any_tracker is None:
                continue
            _pred = CameraTracker._predict_pose(
                _any_tracker.pose_history, frame_ts_ns,
                weight_decay=_weight_decay, vel_ema_rate=_any_tracker.vel_ema,
            )
            if _pred is None:
                continue
            _rv_pred, _tv_pred = _pred
            _R_pred, _ = cv2.Rodrigues(np.asarray(_rv_pred, np.float32).reshape(3, 1))
            _T_cam_ctrl_pred = Transform(
                _R_pred.astype(np.float32), np.asarray(_tv_pred, np.float32).reshape(3)
            )
            _T_world_ctrl_pred = _any_tracker.camera.T_world_cam.compose(_T_cam_ctrl_pred)
            _geom = _any_tracker._geometry

            _occ_per_cam: Dict[int, Tuple[np.ndarray, np.ndarray, object]] = {}
            for cid, cam in self.cameras.items():
                if cam.T_world_cam is None:
                    continue
                _T_cam_occ = cam.T_world_cam.inverse().compose(_T_world_ctrl_pred)
                _occ_per_cam[cid] = (
                    _T_cam_occ.R.astype(np.float32),
                    _T_cam_occ.t.astype(np.float32),
                    _geom,
                )
            occluders_by_ctrl[ctrl_name] = _occ_per_cam

        if occluders_by_ctrl:
            logger.bind(cat="occlusion").debug(
                f"[warm-batch] extrapolated occluders built for: "
                f"{list(occluders_by_ctrl.keys())}"
            )
        return occluders_by_ctrl

    def update_warm_batch(
        self,
        ctrl_names: List[str],
        per_ctrl_observations: Dict[str, Dict[int, np.ndarray]],
        per_ctrl_radii: Optional[Dict[str, Dict[int, np.ndarray]]],
        per_ctrl_brightnesses: Optional[Dict[str, Dict[int, np.ndarray]]],
        frame_ts_ns: int,
    ) -> Dict[str, Optional[Dict]]:
        """Batched cheap-search-only update for controllers that are ALL warm
        this frame (every enabled controller has a prior pose). Submits every
        (ctrl_name, cam_id) cheap-search task across every controller in ONE
        pool round instead of the one-round-per-controller sequence update()
        runs when called once per controller with ctrl_name_filter — this is
        what actually lets two warm controllers' per-camera work overlap.
        Mirrors ControllerTracker.update()'s cheap pass exactly, except for
        the batching and the deliberate absence of blob_mask/claimed_blobs/
        reservations: two controllers matching the same blob is accepted
        here (each controller's own RANSAC/inlier scoring in its independent
        fit absorbs it) — see the warm-warm design notes. No brute-force
        fallback happens inside this path (out of scope for the warm-warm
        case). Cross-controller occlusion IS applied here, but — unlike
        ControllerTracker.update()'s sequential occluder-building, which
        requires another controller's already-confirmed same-frame result
        (structurally incompatible with this batched dispatch, since no
        controller has a same-frame result before the round completes) —
        each controller's occluder pose is its own *extrapolated* prediction
        (see _predict_pose), not a same-frame solve. See
        _build_extrapolated_occluders.

        Returns {ctrl_name: solution_or_None}. None means this controller's
        cheap pass didn't produce a fused solution this frame (e.g. every
        camera came up empty) — the caller must fall back to the existing
        sequential per-controller path (cold re-detect + brute) for that
        controller only, exactly as it would on any other cheap-search
        failure.
        """
        from src.parallel_search import run_cheap_search

        specs = []   # (ctrl_name, cid, tracker, obs, rad, brt)
        for ctrl_name in ctrl_names:
            obs_src = per_ctrl_observations.get(ctrl_name) or {}
            rad_src = (per_ctrl_radii or {}).get(ctrl_name) or {}
            brt_src = (per_ctrl_brightnesses or {}).get(ctrl_name) or {}
            for cid, tracker in self.ctrl_trackers[ctrl_name].trackers.items():
                obs = obs_src.get(cid)
                if obs is None or len(obs) == 0:
                    continue
                specs.append((ctrl_name, cid, tracker, obs, rad_src.get(cid), brt_src.get(cid)))

        if not specs:
            return {c: None for c in ctrl_names}

        occluders_by_ctrl = self._build_extrapolated_occluders(ctrl_names, frame_ts_ns)

        def _other_ctrl(name: str) -> Optional[str]:
            return next((c for c in ctrl_names if c != name), None)

        # Computed ONCE per controller per frame (not once per camera, and
        # not once per controller-camera pair) -- see
        # _predicted_world_for_vel_ema's own docstring.
        _predicted_world_by_ctrl = {
            ctrl_name: _predicted_world_for_vel_ema(
                self.ctrl_trackers[ctrl_name]._fusion_filter,
                self.ctrl_trackers[ctrl_name].trackers.values(), frame_ts_ns)
            for ctrl_name in ctrl_names
        }

        if self._pool is not None:
            futures = {}
            for ctrl_name, cid, tracker, obs, rad, brt in specs:
                prior = {
                    'prev_pose':       tracker.prev_pose,
                    'prev_prev_pose':  tracker.prev_prev_pose,
                    'pose_history':    tracker.pose_history,
                    'vel_ema':         _vel_ema_with_imu_fallback(
                        tracker, _predicted_world_by_ctrl[ctrl_name], self.cameras[cid], frame_ts_ns),
                    'prev_assignment': tracker.prev_assignment,
                    'gyro_rel_R':      _gyro_rel_R_for(self._gyro_data.get(ctrl_name), tracker.pose_history, frame_ts_ns),
                }
                futures[(ctrl_name, cid)] = self._pool.submit(
                    run_cheap_search, (ctrl_name, cid), self._matching_cfg, prior,
                    obs, frame_ts_ns, rad, brt, None,
                    occluders_by_ctrl.get(_other_ctrl(ctrl_name)),
                )
            raw: Dict[Tuple[str, int], Tuple] = {}
            for key, fut in futures.items():
                solution, predicted_pose, norm_prev, norm_prev_prev = fut.result()
                _tracker = self.ctrl_trackers[key[0]].trackers[key[1]]
                _tracker.prev_pose      = norm_prev
                _tracker.prev_prev_pose = norm_prev_prev
                raw[key] = (solution, predicted_pose)
        else:
            raw = {
                (ctrl_name, cid): tracker.search_cheap(
                    obs, frame_ts_ns, blob_radii=rad, blob_brightnesses=brt,
                    occluders_per_cam=occluders_by_ctrl.get(_other_ctrl(ctrl_name)),
                    gyro_rel_R=_gyro_rel_R_for(self._gyro_data.get(ctrl_name), tracker.pose_history, frame_ts_ns),
                    vel_ema_override=_vel_ema_with_imu_fallback(
                        tracker, _predicted_world_by_ctrl[ctrl_name], self.cameras[cid], frame_ts_ns),
                )
                for ctrl_name, cid, tracker, obs, rad, brt in specs
            }

        cam_solutions_per_ctrl: Dict[str, list] = {c: [] for c in ctrl_names}
        eligible_cids_per_ctrl: Dict[str, List[int]] = {c: [] for c in ctrl_names}
        av_count_per_ctrl_cid: Dict[str, Dict[int, int]] = {c: {} for c in ctrl_names}
        for ctrl_name, cid, tracker, obs, rad, brt in specs:
            eligible_cids_per_ctrl[ctrl_name].append(cid)
            av_count_per_ctrl_cid[ctrl_name][cid] = len(obs)
            solution, predicted_pose = raw[(ctrl_name, cid)]
            solution = tracker.finalize_search(
                solution, predicted_pose, obs, blob_radii=rad,
                allow_expensive_fallback=False, frame_ts_ns=frame_ts_ns,
            )
            if solution is not None:
                cam_solutions_per_ctrl[ctrl_name].append(
                    {"cam_id": cid, "tracker": tracker, "solution": solution}
                )

        results: Dict[str, Optional[Dict]] = {}
        for ctrl_name in ctrl_names:
            cam_solutions = cam_solutions_per_ctrl[ctrl_name]
            if not cam_solutions:
                results[ctrl_name] = None
                continue
            # Weak single-camera accept + another eligible camera failed (see
            # _weak_solo_accept_cids' own docstring, and ControllerTracker.update's
            # matching comment for the real recording that motivated this): this
            # batched warm-warm path has NO brute-force of its own (see this
            # method's own docstring), so the only thing to do here is defer --
            # returning None routes this controller through main.py's existing
            # "cheap pass wasn't good enough" fallback (sequential retry, then cold
            # re-detect + real brute-force), exactly as any other cheap-search
            # failure already does, instead of silently accepting an unconfirmed,
            # partially-wrong single-camera solution.
            _weak_solo = _weak_solo_accept_cids(
                cam_solutions, eligible_cids_per_ctrl[ctrl_name], av_count_per_ctrl_cid[ctrl_name],
                self.ctrl_trackers[ctrl_name]._matching_cfg)
            if _weak_solo:
                logger.bind(cat="matching_decisions").debug(
                    f"[{ctrl_name}] weak solo accept in warm-batch (cam(s) "
                    f"{sorted({cs['cam_id'] for cs in cam_solutions})}) while cam(s) {_weak_solo} "
                    f"had blobs but no accepted solution -- deferring to sequential brute fallback"
                )
                results[ctrl_name] = None
                continue
            obs_src = per_ctrl_observations.get(ctrl_name) or {}
            results[ctrl_name] = self.ctrl_trackers[ctrl_name]._fuse_and_finalize(
                cam_solutions, obs_src, claimed_blobs=None, frame_ts_ns=frame_ts_ns,
                fixed_primary_cam=self._fixed_primary_cam, self_cal=self._self_cal,
            )
        return results

    def update_cold_batch(
        self,
        ctrl_names: List[str],
        per_ctrl_observations: Dict[str, Dict[int, np.ndarray]],
        per_ctrl_radii: Optional[Dict[str, Dict[int, np.ndarray]]],
        per_ctrl_brightnesses: Optional[Dict[str, Dict[int, np.ndarray]]],
        frame_ts_ns: int,
        committed_solutions: Optional[Dict[str, Dict]] = None,
        committed_observations: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        committed_radii: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    ) -> Dict[str, Optional[Dict]]:
        """Batched brute-force-only update for controllers that are ALL truly
        cold this frame (every camera tracker's prev_pose is None — callers
        must only pass controllers satisfying that; main.py guarantees it by
        routing only ctrl_has_prior[name] is False controllers here). Submits
        every (ctrl_name, cam_id) tier-round brute-force task across every
        controller in ONE shared set of pool rounds, instead of the
        one-controller-at-a-time sequence main.py's per-controller
        _update_ctrl(force_brute=True) loop runs today — this is what lets
        two simultaneously-cold controllers' brute recovery overlap. Mirrors
        ControllerTracker.update()'s tier-round brute path (see there), with
        two structural differences:

          - The tier-widening stop condition ("any camera hit strong_found
            this round -> stop widening") is evaluated per (ctrl_name, cid)
            key via `not st.strong_found`, not per-controller-with-an-early-
            break -- so one controller reaching strong_found on an early tier
            does not truncate another controller's later tiers, and vice
            versa. Both controllers' cameras simply share the same round of
            pool.submit calls.

          - Since no controller here has any prior pose or pose_history (by
            definition -- see above), there is no claimed_blobs exclusion, no
            reservation biasing, and no occluders_per_cam during the search
            itself (same reasoning as _build_extrapolated_occluders: occlusion
            needs a pose to project from, and none exists yet for any of
            these controllers). Every candidate is computed independently and
            only reconciled with the others AFTER all of them finish solving
            -- see _resolve_cold_conflicts. Two cold controllers matching the
            same blob, or landing in a mutually impossible (occluding)
            configuration, are real possibilities here in a way they aren't
            for update_warm_batch's warm priors, precisely because nothing
            here excludes the other controller's candidate blobs/pose during
            the search.

        committed_solutions: {ctrl_name: solution} for controllers already
        committed to tracker state THIS frame before update_cold_batch was
        called -- i.e. every warm-processed controller (get_ctrl_processing_
        order guarantees these always precede the cold suffix, so "already
        committed" and "warm this frame" coincide in practice). These are
        never returned or re-solved here; they're folded into
        _resolve_cold_conflicts purely as FIXED candidates a cold candidate
        can lose to but never beat -- closing the gap where a cold
        controller's brute search could otherwise land on a pose that
        shares a blob with, or geometrically occludes, an already-tracked
        warm controller with nothing to catch it (unlike warm-warm, which
        gets occlusion-awareness from _build_extrapolated_occluders before
        the search even runs, and unlike cold-cold, which this method
        already reconciles post-hoc). Omit or pass {} for no such check.

        committed_observations / committed_radii: {ctrl_name: {cam_id:
        centroids/radii}} for the SAME already-committed controllers named
        in committed_solutions -- their own per-controller blob arrays, in
        the index space committed_solutions' assignment/aux_assignments blob
        indices refer to. Required to check a committed candidate for a
        genuine shared-blob conflict against these cold candidates: main.py
        detects blobs per controller (each with its own warm/cold ROI), so a
        committed controller's blob array is generally NOT the same array
        (or index space) as any cold controller's here, and comparing raw
        indices without this geometry would produce false-positive conflicts
        purely from small indices coincidentally recurring in both
        independent arrays (see _resolve_cold_conflicts' blob_geometry
        docstring). Omit to skip the shared-blob check against committed
        candidates entirely (cross-occlusion still applies).

        Returns {ctrl_name: solution_or_None}. None means either this
        controller's brute search didn't produce a solution this frame (no
        camera cleared the K-frame persistence gate, or none solved), or it
        did solve but lost a same-frame conflict -- against another cold
        controller's better-fit candidate, or against an already-committed
        warm controller it can never outrank -- both cases are routed
        through _mark_all_lost, identically to any other failed brute
        search, so _consecutive_good_blob_frames is left untouched either
        way (see the conflict-loser branch below for why that matters).
        """
        from src.parallel_search import run_brute_tier

        _confirm_frames = int(self._matching_cfg.get('cold_brute_force_confirm_frames', 3))
        states: Dict[Tuple[str, int], object] = {}
        _other_cams_by_key: Dict[Tuple[str, int], List[Tuple]] = {}
        for ctrl_name in ctrl_names:
            obs_src = per_ctrl_observations.get(ctrl_name) or {}
            for cid, tracker in self.ctrl_trackers[ctrl_name].trackers.items():
                obs_full = obs_src.get(cid)
                if obs_full is None or len(obs_full) == 0:
                    continue
                _other_cams_by_key[(ctrl_name, cid)] = _build_other_cameras_blobs(
                    self.cameras, obs_src, cid)
                # Persistence gate, verbatim logic from ControllerTracker.update()
                # (controller.py, this file) -- every controller passed to
                # update_cold_batch is already known cold by the caller's contract
                # (no camera has a usable prior right now). No _ever_tracked
                # exemption here either (removed 2026-09-11, see
                # ControllerTracker.update()'s own copy of this check for the
                # full reasoning) -- a controller confirmed real earlier this
                # session still needs each individual reacquisition candidate
                # to clear the same noise-persistence bar as a first-time one.
                if len(obs_full) >= tracker._pose_searcher._c_brute_min_inliers:
                    tracker._consecutive_good_blob_frames += 1
                else:
                    tracker._consecutive_good_blob_frames = 0
                if tracker._consecutive_good_blob_frames < _confirm_frames:
                    logger.bind(cat="matching_decisions").debug(
                        f"[{ctrl_name} | cam {cid}] cold brute-force gated (batch): "
                        f"streak={tracker._consecutive_good_blob_frames}/{_confirm_frames} "
                        f"(n_blobs={len(obs_full)})"
                    )
                    continue
                mask = np.ones(len(obs_full), dtype=bool)
                states[(ctrl_name, cid)] = tracker._pose_searcher.new_brute_state(
                    obs_full, pose_prior=None, other_cameras_blobs=_other_cams_by_key[(ctrl_name, cid)],
                    blob_mask=mask, occluders_per_cam=None,
                )

        if not states:
            # No camera of any controller here even cleared the confirm-frames gate
            # (or had any blobs at all) -- nothing to run brute-force against, but
            # every controller still needs its own _mark_all_lost bookkeeping this
            # frame (consecutive_failures, prev_pose/pose_history grace-expiry, and
            # crucially should_force_cold_start's elapsed-time/consecutive-rejects
            # check on the fusion filter). The early return below used to skip this
            # entirely, silently freezing should_force_cold_start's clock: a
            # controller that ran dry on blobs for a stretch (a real occlusion, or
            # this recording's "static_dark" low light) never got its fusion filter
            # reset, so it kept coasting on a contaminated pre-loss prediction
            # indefinitely instead of ever being re-declared cold (found
            # investigating a report of a controller "freezing" in place well past
            # when tracking should have gone lost -- the OTHER call sites of
            # _mark_all_lost below this one, for a controller that DOES clear the
            # gate but then fails to solve, were never affected).
            for ctrl_name in ctrl_names:
                self.ctrl_trackers[ctrl_name]._mark_all_lost(frame_ts_ns)
            return {c: None for c in ctrl_names}

        _any_ps = next(
            (self.ctrl_trackers[k[0]].trackers[k[1]]._pose_searcher
             for k, st in states.items() if st is not None),
            None,
        )
        max_tiers = len(_any_ps._c_brute_depth_tiers) if _any_ps is not None else 0

        if self._pool is not None:
            for tier_idx in range(max_tiers):
                remaining = [k for k, st in states.items() if st is not None and not st.strong_found]
                if not remaining:
                    break
                logger.bind(cat="batch_orchestration").debug(f"[cold-batch] tier {tier_idx}: submitting {sorted(remaining)}")
                futures = {k: self._pool.submit(run_brute_tier, k, states[k], tier_idx) for k in remaining}
                for k, fut in futures.items():
                    states[k] = fut.result()
        else:
            for tier_idx in range(max_tiers):
                remaining = [k for k, st in states.items() if st is not None and not st.strong_found]
                if not remaining:
                    break
                for ctrl_name, cid in remaining:
                    self.ctrl_trackers[ctrl_name].trackers[cid]._pose_searcher.brute_search_tier(
                        states[(ctrl_name, cid)], tier_idx)

        cam_solutions_per_ctrl: Dict[str, list] = {c: [] for c in ctrl_names}
        for (ctrl_name, cid), st in states.items():
            if st is None:
                continue
            tracker = self.ctrl_trackers[ctrl_name].trackers[cid]
            sol = tracker._pose_searcher.finalize_brute_state(st)
            if sol is None:
                continue
            obs_full = (per_ctrl_observations.get(ctrl_name) or {})[cid]
            rad_full = (per_ctrl_radii or {}).get(ctrl_name, {}).get(cid)
            mask = np.ones(len(obs_full), dtype=bool)
            sol = tracker.finalize_search(
                sol, None, obs_full, blob_radii=rad_full,
                other_cameras_blobs=_other_cams_by_key[(ctrl_name, cid)],
                blob_mask=mask, occluders_per_cam=None, allow_expensive_fallback=True,
                frame_ts_ns=frame_ts_ns,
            )
            if sol is not None:
                cam_solutions_per_ctrl[ctrl_name].append(
                    {"cam_id": cid, "tracker": tracker, "solution": sol}
                )

        _min_inliers = float(self._matching_cfg.get('min_inliers', 4))
        _error_floor = float(self._matching_cfg.get(
            'score_error_floor_px', self._matching_cfg.get('strong_match_error_px', 0.5)))
        candidates: Dict[str, Dict] = {}
        cam_solutions_of: Dict[str, list] = {}
        for ctrl_name in ctrl_names:
            cam_solutions = cam_solutions_per_ctrl[ctrl_name]
            if not cam_solutions:
                self.ctrl_trackers[ctrl_name]._mark_all_lost(frame_ts_ns)
                continue
            # Cold-cold candidates never excluded each other's blobs during search
            # (see this method's docstring) -- two of THIS controller's own cameras
            # can each independently land on their own self-consistent-looking pose,
            # one of them built on a blob that actually belongs to another
            # controller. Blindly joint-fusing both would let the contaminated one
            # poison the result (see _select_best_cam_solution); pick a single
            # cross-camera-validated winner instead of blending.
            cam_solutions = _select_best_cam_solution(cam_solutions, _min_inliers, _error_floor)
            obs_src = per_ctrl_observations.get(ctrl_name) or {}
            candidates[ctrl_name] = self.ctrl_trackers[ctrl_name]._compute_fused_solution(
                cam_solutions, obs_src, fixed_primary_cam=self._fixed_primary_cam,
            )
            cam_solutions_of[ctrl_name] = cam_solutions

        if not candidates:
            return {c: None for c in ctrl_names}

        _committed = committed_solutions or {}
        _all_for_conflict_check = {**candidates, **_committed}
        _blob_geometry: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
        for ctrl_name in ctrl_names:
            _cents = per_ctrl_observations.get(ctrl_name) or {}
            _radii = (per_ctrl_radii or {}).get(ctrl_name) or {}
            _blob_geometry[ctrl_name] = {
                cid: (_cents[cid], _radii[cid]) for cid in _cents if cid in _radii
            }
        for ctrl_name in _committed:
            _cents = (committed_observations or {}).get(ctrl_name) or {}
            _radii = (committed_radii or {}).get(ctrl_name) or {}
            _blob_geometry[ctrl_name] = {
                cid: (_cents[cid], _radii[cid]) for cid in _cents if cid in _radii
            }
        losers, loser_reason = self._resolve_cold_conflicts(
            _all_for_conflict_check, fixed_names=set(_committed.keys()),
            blob_geometry=_blob_geometry,
        )

        results: Dict[str, Optional[Dict]] = {}
        for ctrl_name, solution in candidates.items():
            if ctrl_name in losers:
                # Same bookkeeping as any other failed brute search
                # (_mark_all_lost, controller.py:767-789) -- deliberately NOT
                # a fresh reset of _consecutive_good_blob_frames. This
                # controller already cleared the K-frame gate above, before
                # any brute enumeration even ran, so it does not need to
                # re-accumulate _confirm_frames consecutive frames before
                # retrying next frame -- only _mark_all_lost's own fields
                # (consecutive_failures, tracking_lost_last_frame) change.
                logger.bind(cat="occlusion").info(
                    f"[cold-batch] conflict: dropping {ctrl_name} — "
                    f"{loser_reason.get(ctrl_name, 'lost to a better inlier-discounted candidate this frame')}"
                )
                self.ctrl_trackers[ctrl_name]._mark_all_lost(frame_ts_ns)
                results[ctrl_name] = None
                continue
            obs_src = per_ctrl_observations.get(ctrl_name) or {}
            self.ctrl_trackers[ctrl_name]._commit_fused_solution(
                solution, cam_solutions_of[ctrl_name], obs_src,
                claimed_blobs=None, frame_ts_ns=frame_ts_ns, self_cal=self._self_cal,
            )
            results[ctrl_name] = solution

        for ctrl_name in ctrl_names:
            results.setdefault(ctrl_name, None)
        return results

    def _resolve_cold_conflicts(
        self, candidates: Dict[str, Dict], fixed_names: Optional[Set[str]] = None,
        blob_geometry: Optional[Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]] = None,
        blob_match_margin_px: float = 5.0,
    ) -> Tuple[Set[str], Dict[str, str]]:
        """Detect and greedily resolve conflicts among simultaneously-solved
        cold-start candidates (update_cold_batch, above) BEFORE any of them
        is committed to tracker state.

        fixed_names: candidates (typically already-committed warm controllers
        passed in via update_cold_batch's committed_solutions) that must
        never appear in the returned loser set -- their tracker state is
        already committed and un-committing isn't supported, so they always
        win any conflict they're part of. Processed before the normal greedy
        pass below (see the loop that drains fixed_names first). Omit for the
        pure cold-cold case -- behavior is then identical to before this
        parameter existed.

        blob_geometry: {ctrl_name: {cam_id: (centroids, radii)}} -- each
        controller's OWN per-camera blob arrays, in the same index space its
        solution's `assignment`/`aux_assignments` blob indices refer to.
        Required for the shared-blob check to mean anything: main.py runs
        blob detection PER CONTROLLER (each controller's warm/cold ROI
        differs), so ctrl_a's blob index 5 in cam0 and ctrl_b's blob index 5
        in cam0 are, in general, indices into two completely independent
        arrays -- not the same physical blob -- UNLESS that (controller,
        camera) pair was deduped onto a single shared detect() call (see
        _run_blob_detect_batch_multi's dedup grouping in main.py), which is
        the only case where index equality alone would happen to be valid.
        Comparing raw indices unconditionally (the original implementation)
        produced false "shared blob" conflicts any time both candidates
        simply had low indices in their own independent arrays -- e.g. two
        7-blob candidates sharing indices [0..6] in cam0 even though not one
        of those index pairs was the same physical blob. Passing
        blob_geometry fixes this: a shared blob is now decided by whether
        the two candidates' matched blobs' PIXEL POSITIONS actually overlap
        (centroid distance below the sum of their radii + blob_match_margin_px),
        mirroring the same-camera same-frame blob exclusion main.py already
        does elsewhere (_exclude_claimed_blobs, main.py) for exactly this
        reason. If blob_geometry is omitted, or a (ctrl_name, cam_id) pair is
        missing from it, the shared-blob check is skipped for that pair
        entirely (never flagged) rather than risk a false positive from
        coincidentally-equal local indices -- cross-occlusion (below, pose-
        geometry based, unaffected by this issue) remains the catch-all for
        real cross-controller conflicts in that case.

        Two candidates conflict if ANY of:
          - physical overlap: their fused T_world_ctrl centers are closer
            than min_controller_center_distance_m -- two rigid controllers
            cannot occupy overlapping 3D space. Camera-agnostic: catches a
            bad candidate even when it shares no registered camera with the
            other at all. Always checked, regardless of _occlusion_on.
          - shared blob: their matched blobs' pixel centroids, in the same
            camera, overlap within blob_match_margin_px (see blob_geometry
            above) -- primary `assignment` or `aux_assignments` on either
            side.
          - cross-occlusion: one candidate's SOLVED pose (both here -- unlike
            _build_extrapolated_occluders, which necessarily uses a
            *predicted* pose since neither warm-warm controller has solved
            yet when it runs) should, per _cross_occluded_mask, have blocked
            one of the other candidate's matched LEDs from view in a camera
            they both used. Checked in both directions.
          - aux-projection collision: one candidate's solved pose, reprojected
            into a camera where the OTHER candidate has a registered match
            (primary or aux), lands on that other candidate's actual claimed
            blobs there -- checked even if the projecting candidate's own
            search never registered that camera as one of its own matches
            (unlike the shared-blob check, which requires both sides to have
            already matched the same camera). Checked in both directions.

        All conflict types are treated identically -- no special-casing, no
        cheap-repair exception: any conflict of any kind drops the loser
        outright. Winner selection is greedy: repeatedly take the remaining
        candidate with the lowest inlier-discounted error (see _score below;
        ties broken by more total matched pairs across assignment +
        aux_assignments), keep it, and drop every candidate directly in
        conflict with it; repeat among what's left. This correctly handles
        conflict components of 3+ controllers, not just pairs -- a controller
        connected to the graph only through an already-dropped neighbor is
        free to win in a later round. Note this scoring is what actually
        resolves an aux-projection collision or physical overlap in practice:
        the candidate with more total corroborated inliers (primary + aux)
        and lower combined error wins, so a locally-good-but-uncorroborated
        fit loses to a well-corroborated one without any extra logic.

        Returns (losers, loser_reason): the set of ctrl_names to drop, and a
        {ctrl_name: human-readable reason} map for every dropped name —
        which winner it lost to and which of the four conflict types fired
        (with camera + blob/LED indices where applicable).
        candidates values must be solution dicts as produced by
        ControllerTracker._compute_fused_solution (primary_cam, T_world_ctrl,
        error, assignment, aux_assignments).
        """
        names = list(candidates.keys())
        if len(names) < 2:
            return set(), {}

        def _matched_pairs(sol: Dict, cid: int) -> List[Tuple[int, int]]:
            if sol.get('primary_cam') == cid:
                return sol.get('assignment') or []
            return (sol.get('aux_assignments') or {}).get(cid) or []

        def _matched_cams(sol: Dict) -> Set[int]:
            cams = set((sol.get('aux_assignments') or {}).keys())
            if sol.get('primary_cam') is not None:
                cams.add(sol['primary_cam'])
            return cams

        _occlusion_on    = bool(self._matching_cfg.get('cross_controller_occlusion', False))
        _br              = float(self._matching_cfg.get('cross_occlusion_bounding_radius_m', 0.18))
        _gate_margin     = float(self._matching_cfg.get('cross_occlusion_gate_margin_px', 20.0))
        _min_center_dist = float(self._matching_cfg.get('min_controller_center_distance_m', 0.05))

        # Human-readable explanation for each conflicting pair, keyed by
        # frozenset({a, b}) — filled in below as each conflict is found, so
        # the drop-site log (update_cold_batch) can report exactly which
        # blob(s) collided or which controller's body occluded which LED(s),
        # instead of just "lost to a better candidate".
        conflict_reason: Dict[FrozenSet[str], str] = {}

        conflicts: Dict[str, Set[str]] = {n: set() for n in names}
        for i, a in enumerate(names):
            sol_a = candidates[a]
            for b in names[i + 1:]:
                sol_b = candidates[b]
                conflict = False

                # Two rigid controllers cannot occupy overlapping 3D space --
                # camera-agnostic, so it catches a bad candidate even when it
                # shares no registered camera with the other (e.g. two cold
                # cameras solved for two different controllers but both
                # actually recovered the same physical controller's pose).
                if _min_center_dist > 0.0:
                    center_dist = float(np.linalg.norm(
                        sol_a['T_world_ctrl'].t - sol_b['T_world_ctrl'].t))
                    if center_dist < _min_center_dist:
                        conflict = True
                        conflict_reason[frozenset((a, b))] = (
                            f"physical overlap: centers {center_dist * 100:.1f}cm apart "
                            f"(min {_min_center_dist * 100:.1f}cm)"
                        )

                for cid in (_matched_cams(sol_a) & _matched_cams(sol_b)) if not conflict else ():
                    pairs_a = _matched_pairs(sol_a, cid)
                    pairs_b = _matched_pairs(sol_b, cid)
                    if not pairs_a or not pairs_b:
                        continue
                    geo_a = (blob_geometry or {}).get(a, {}).get(cid)
                    geo_b = (blob_geometry or {}).get(b, {}).get(cid)
                    if geo_a is None or geo_b is None:
                        # No blob geometry for this (ctrl, cam) pair -- can't
                        # tell whether these two candidates' blob indices
                        # refer to the same physical blob or just happen to
                        # collide numerically in two independent per-
                        # controller arrays (see docstring). Skip rather than
                        # risk a false positive.
                        continue
                    cent_a, rad_a = geo_a
                    cent_b, rad_b = geo_b
                    idx_a = np.array([blob for blob, _ in pairs_a], dtype=int)
                    idx_b = np.array([blob for blob, _ in pairs_b], dtype=int)
                    pts_a, pts_b = cent_a[idx_a], cent_b[idx_b]
                    rr_a,  rr_b  = rad_a[idx_a],  rad_b[idx_b]
                    dists = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=2)
                    overlap = dists < (rr_a[:, None] + rr_b[None, :] + blob_match_margin_px)
                    if overlap.any():
                        conflict = True
                        _ia, _ib = np.where(overlap)
                        conflict_reason[frozenset((a, b))] = (
                            f"shared blob in cam{cid}: {a}#{int(idx_a[_ia[0]])} "
                            f"~= {b}#{int(idx_b[_ib[0]])} (dist={dists[_ia[0], _ib[0]]:.1f}px)"
                        )
                        break

                if not conflict and _occlusion_on:
                    for occluder, victim, sol_occ, sol_vic in ((a, b, sol_a, sol_b), (b, a, sol_b, sol_a)):
                        tracker_vic = next(iter(self.ctrl_trackers[victim].trackers.values()))
                        geom_occ    = next(iter(self.ctrl_trackers[occluder].trackers.values()))._geometry
                        positions_vic = tracker_vic.model.positions
                        for cid in _matched_cams(sol_vic):
                            cam = self.cameras.get(cid)
                            if cam is None or cam.T_world_cam is None:
                                continue
                            led_ids = [lid for _, lid in _matched_pairs(sol_vic, cid)]
                            if not led_ids:
                                continue
                            T_cam_vic = cam.T_world_cam.inverse().compose(sol_vic['T_world_ctrl'])
                            T_cam_occ = cam.T_world_cam.inverse().compose(sol_occ['T_world_ctrl'])
                            focal_px = float(max(cam.camera_matrix[0, 0], cam.camera_matrix[1, 1]))
                            occluded = _cross_occluded_mask(
                                T_cam_vic.R.astype(np.float32), T_cam_vic.t.astype(np.float32),
                                positions_vic,
                                T_cam_occ.R.astype(np.float32), T_cam_occ.t.astype(np.float32),
                                geom_occ, _br, _br, focal_px, _gate_margin,
                                log_tag=f"[cold-conflict {occluder}->{victim} cam{cid}]",
                            )
                            _hit_leds = [lid for lid in led_ids if occluded[lid]]
                            if _hit_leds:
                                conflict = True
                                conflict_reason[frozenset((a, b))] = (
                                    f"cross-occlusion: {occluder}'s body blocks {victim}'s "
                                    f"LED(s) {sorted(_hit_leds)} in cam{cid}"
                                )
                                break
                        if conflict:
                            break

                # One candidate's solved pose, reprojected into a camera where
                # the OTHER candidate has a registered match (primary or aux),
                # lands on that other candidate's actual claimed blobs -- a
                # direct physical-collision signal that doesn't depend on
                # `proj`'s own search having registered an aux hit in that
                # camera itself (unlike the shared-blob check above, which
                # only fires when BOTH sides already matched the same
                # camera). Resolution still goes through the same
                # inlier-discounted `_score` below, so whichever candidate
                # has more total corroborated inliers (primary + aux) and
                # lower combined error naturally wins.
                if not conflict and _occlusion_on:
                    for proj_name, proj_sol, main_name, main_sol in (
                        (a, sol_a, b, sol_b), (b, sol_b, a, sol_a),
                    ):
                        tracker_proj = next(iter(self.ctrl_trackers[proj_name].trackers.values()))
                        for cid in _matched_cams(main_sol):
                            main_pairs = _matched_pairs(main_sol, cid)
                            if not main_pairs:
                                continue
                            cam = self.cameras.get(cid)
                            geo_main = (blob_geometry or {}).get(main_name, {}).get(cid)
                            if cam is None or cam.T_world_cam is None or geo_main is None:
                                continue
                            cent_main, rad_main = geo_main
                            idx_main = np.array([blob for blob, _ in main_pairs], dtype=int)
                            pts_main, rr_main = cent_main[idx_main], rad_main[idx_main]

                            T_ci = cam.T_world_cam.inverse().compose(proj_sol['T_world_ctrl'])
                            R_p, t_p = T_ci.R.astype(np.float32), T_ci.t.astype(np.float32)
                            vis_ids = np.where(_visible_mask(
                                R_p, t_p, tracker_proj.model.positions, tracker_proj.model.normals,
                                tracker_proj._geometry, cam_K=cam.camera_matrix, cam_dc=cam.dist_coeffs,
                                cam_w=cam.width, cam_h=cam.height, cam_rpmax=cam.rpmax,
                                cam_is_fisheye=cam.is_fisheye,
                            ) >= 1.0)[0]
                            if len(vis_ids) == 0:
                                continue
                            proj_pts = _project_points(
                                cv2.Rodrigues(R_p)[0], t_p, tracker_proj.model.positions[vis_ids],
                                cam.camera_matrix, cam.dist_coeffs, is_fisheye=cam.is_fisheye,
                            )
                            dists = np.linalg.norm(proj_pts[:, None, :] - pts_main[None, :, :], axis=2)
                            overlap = dists < (rr_main[None, :] + blob_match_margin_px)
                            if overlap.any():
                                conflict = True
                                _iv, _im = np.where(overlap)
                                conflict_reason[frozenset((a, b))] = (
                                    f"aux-projection collision in cam{cid}: {proj_name}'s LED"
                                    f"{int(vis_ids[_iv[0]])} projects onto {main_name}'s claimed "
                                    f"blob #{int(idx_main[_im[0]])} (dist={dists[_iv[0], _im[0]]:.1f}px)"
                                )
                                break
                        if conflict:
                            break

                if conflict:
                    conflicts[a].add(b)
                    conflicts[b].add(a)

        if not any(conflicts.values()):
            return set(), {}

        # A fit sitting right at the minimal-inlier floor has almost no spare
        # degrees of freedom, so a near-zero residual there is not evidence of
        # a correct match -- it's just what an under-constrained fit looks
        # like (compare _redundancy_factor in pose_search.py's proximity
        # path, same reasoning). Comparing raw error across candidates with
        # very different inlier counts lets exactly that kind of fit
        # outrank a genuinely well-constrained, higher-coverage, multi-camera
        # one. Discount error linearly by how many multiples of the floor a
        # candidate has: right at the floor -> no discount; 5x the floor ->
        # error divided by 5.
        #
        # error_floor additionally clamps the error itself before that
        # discount: two candidates at 0.06px and 0.09px are not distinguishable
        # fits -- that gap is blob-centroid detection noise, not evidence one
        # pose is really better -- so letting the discount formula's raw
        # multiplicative ratio decide between them would let sub-pixel noise
        # outrank a candidate with genuinely more corroborating total_pairs.
        # Clamping both to this floor first means only a genuine, above-floor
        # error difference can still decide the score; two comparably-good
        # fits fall through to being decided by total_pairs alone.
        _min_inliers = float(self._matching_cfg.get('min_inliers', 4))
        _error_floor = float(self._matching_cfg.get(
            'score_error_floor_px', self._matching_cfg.get('strong_match_error_px', 0.5)))

        def _score(name: str) -> Tuple[float, int]:
            sol = candidates[name]
            total_pairs = len(sol.get('assignment') or []) + sum(
                len(v) for v in (sol.get('aux_assignments') or {}).values()
            )
            effective_error = _inlier_discounted_error(sol['error'], total_pairs, _min_inliers, _error_floor)
            return (effective_error, -total_pairs)

        def _fmt(name: str) -> str:
            """Human-readable evidence summary for one side of a conflict --
            raw error, primary vs. aux inlier counts, and the same
            inlier-discounted score _score/the greedy loop actually decide
            on, so the log line at the drop site (update_cold_batch) shows
            exactly why the winner outranked the loser, not just that it did."""
            sol = candidates[name]
            n_primary = len(sol.get('assignment') or [])
            n_aux = sum(len(v) for v in (sol.get('aux_assignments') or {}).values())
            effective_error, _ = _score(name)
            return (
                f"err={sol['error']:.2f}px primary={n_primary} aux={n_aux} "
                f"total={n_primary + n_aux} score={effective_error:.3f}"
            )

        remaining = set(names)
        losers: Set[str] = set()
        loser_reason: Dict[str, str] = {}

        # Fixed candidates always win — drain them first, unconditionally,
        # before the score-based greedy pass below ever runs. A candidate
        # conflicting with a fixed one is dropped here regardless of its own
        # error, and a fixed candidate is never itself added to losers —
        # including in the (out of scope for this method to resolve, since
        # neither side can be un-committed) case where two fixed candidates
        # conflict with each other: that conflict is simply left unresolved
        # rather than corrupting the loser set.
        _fixed = set(fixed_names or ())
        for fixed in _fixed & remaining:
            if fixed not in remaining:
                continue
            remaining.discard(fixed)
            for loser in conflicts[fixed] & remaining:
                if loser in _fixed:
                    continue
                losers.add(loser)
                _reason = conflict_reason.get(frozenset((fixed, loser)), "unknown")
                loser_reason[loser] = (
                    f"lost to already-committed {fixed} | "
                    f"winner[{fixed}]: {_fmt(fixed)} | loser[{loser}]: {_fmt(loser)} | "
                    f"reason: {_reason}"
                )
                remaining.discard(loser)

        while remaining:
            winner = min(remaining, key=_score)
            remaining.discard(winner)
            for loser in conflicts[winner] & remaining:
                losers.add(loser)
                _reason = conflict_reason.get(frozenset((winner, loser)), "unknown")
                loser_reason[loser] = (
                    f"lost to {winner} | "
                    f"winner[{winner}]: {_fmt(winner)} | loser[{loser}]: {_fmt(loser)} | "
                    f"reason: {_reason}"
                )
                remaining.discard(loser)
        return losers, loser_reason
