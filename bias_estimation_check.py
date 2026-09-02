#!/usr/bin/env python3
"""
bias_estimation_check.py -- Step 4 of the IMU+vision fusion plan: a joint
batch least-squares solve for per-node velocity and slowly-varying IMU bias
(accel + gyro), against FIXED vision reference nodes (world_vision_poses --
the same UN-bridged world-frame poses gyro_preint_check.py/
accel_short_horizon_check.py already use).

Motivation (see memory project_imu_fusion_step3_accel.md, Findings 1-10):
Step 3 found short-horizon accel dead-reckoning with a fixed b_a=0 loses to
naive constant-velocity extrapolation. External corroboration (a CV1 sensor-
fusion developer, Finding 10) says this is the expected outcome UNTIL IMU
bias is estimated online rather than assumed zero -- this script builds that
missing piece as a single offline batch solve (not a real-time filter),
using scipy.optimize.least_squares the same way src/_self_calibration.py and
src/pose_search.py already do.

State per reference node k: velocity v_k (3,), accel bias b_a_k (3,), gyro
bias b_g_k (3,) -- 9 unknowns/node -- PLUS one GLOBAL (not per-node) 3-DOF
rotation delta rot_delta, packed as
x = [v_0..v_N-1, b_a_0..b_a_N-1, b_g_0..b_g_N-1, rot_delta]. Vision-measured
position/orientation per node are held FIXED, not part of the optimized
state (see plan's "Scope for this pass (v1)").

rot_delta (added 2026-09-02, see visualization/controller_calibration_for_basalt/
README.md findings 7-8): an axis-angle CORRECTION on top of _DIAG_FLIP
(diag(1,-1,-1), a precise 180deg flip about X) -- load_and_calibrate_
controller_imu now applies _DIAG_FLIP too (as of this same date), but this
script still can't just call it directly: rot_delta needs to solve for a
correction ON TOP of _DIAG_FLIP, so it needs the raw, pre-axis-transform
stream to apply that correction to, not load_and_calibrate_controller_imu's
already-fixed output. Cross-session diagnostic testing found the OLD
_Y_FLIP @ Rt.R^(±1) transform (superseded, see src/imu_data.py's module
docstring) produced gyro/accel residuals 34-150 sigma out even with bias
and lever arm otherwise correctly modeled, while swapping in _DIAG_FLIP
directly cut r_gyro by 53-76% -- but not to zero, and asymmetrically between
controllers
(left improves far more than right) -- so this is a single SHARED (same
rotation applied to both accel and gyro, per the "one physical chip package,
one mounting orientation" hypothesis _DIAG_FLIP's own discovery suggests)
GLOBAL correction for this solve to refine per controller, initialized at
rot_delta=0 (i.e. exactly _DIAG_FLIP), rather than trusting either session's
externally-composed estimate. It is NOT per-node: unlike bias (which
genuinely drifts) this is a fixed hardware/calibration property for the
whole recording, same treatment as the lever arm.

Residuals per consecutive node pair (k, k+1), all built from already-
existing src/imu_data.py math:
  - gyro rotation                 (gyro_preint_residual)
  - accel velocity + position     (accel_preint_residual, new)
  - gyro/accel bias random walk   (both functions' r_bias output)
Plus one anchor residual on node 0's b_a/b_g, pinning it toward zero --
added per Finding 11 (a real-time reference fusion implementation read in
full 2026-09-01, github.com/Beyley/monado's sensor_fusion.cpp). Without an
anchor, the bias random-walk chain has a gauge freedom: every b_a_k could
shift by the same constant with zero cost to any random-walk residual,
since those only see DIFFERENCES between consecutive nodes -- "without
marginalization IMU becomes unconstrained except by observation" in
Beyley's own words for the equivalent problem in his real-time solve.

Residual weighting: every residual is divided by a rough per-family sigma so
the concatenated vector fed to least_squares is roughly "in standard
deviations" (same spirit as Beyley's reference code, which sizes its own
Huber threshold in sigma units rather than raw pixels) -- gyro rotation by
factory gyro noise_std (rad/s) * sqrt(dt) (integrated-white-noise scaling
for a single integration), accel velocity/position by factory accel
noise_std similarly scaled (sqrt(dt) for velocity, dt*sqrt(dt) for the
double-integrated position). This reuses the only real per-axis noise
numbers this project has -- ControllerImuAxisCalib.noise_std, loaded from
the factory JSON's Noise field but unused until now -- rather than inventing
arbitrary weights, but the dt-scaling drops exact constant factors (a rough
engineering scale, not a rigorously propagated covariance). Revisit if the
solve's behavior suggests one residual family is dominating disproportion-
ately. Bias random-walk/anchor sigmas reuse bias_uncertainty the same way
(also loaded from the factory JSON, also previously unused).

Usage: python bias_estimation_check.py [path/to/config.yml] [output_dir] [window_size] [max_nfev]
window_size (default 300, "a few hundred consecutive accepted frames" per
the Step 4 plan) is deliberately NOT the full recording -- see the plan's
"Scale/windowing" note: state size grows as 9*window_size. The plan expected
jac_sparsity to only matter at full-recording scale; empirically (measured
while implementing this script) a dense finite-difference Jacobian was
already impractical at a 30-node/270-unknown window (~271 residual_fn calls
per solver iteration, each iteration several seconds), so jac_sparsity
(_numeric_sparsity below) is used from v1, not deferred. max_nfev
(default 2000) is scipy's own cap but is NOT a reliable wall-clock bound on
its own for this problem shape -- see _numeric_sparsity's docstring.
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from accel_short_horizon_check import low_motion_bootstrap_g_world
from accel_sign_check import _MAX_PAIR_DT_S, world_vision_poses
from compare_vision_mocap import load_pose_csv, load_device_mocap
from src.imu_data import (accel_preint_residual, create_imu_calib_from_config, gyro_preint_residual, load_imu_csv,
                           slice_imu_to_window)
from src.load_config import load_json_config, load_yaml_config

_IMU_FILES = {"left_controller":  ("imu1/data.csv", -5_000_000),
              "right_controller": ("imu2/data.csv", -7_000_000)}
_AXIS_NAMES = ("x", "y", "z")
_DEFAULT_WINDOW = 300  # "a few hundred consecutive accepted frames" per the Step 4 plan
_ANCHOR_SIGMA_SCALE = 1.0  # anchor sigma = bias_uncertainty * this -- see module docstring
_MIN_SIGMA = 1e-9  # floor so a zero/missing factory noise/uncertainty field can't divide by zero

# The first real 80-node solve (this diagnostic session) DID NOT CONVERGE with residual RMS
# stuck at 73-86 sigma (barely moving over 2000 evals). A residual_fn(x0) breakdown found ALL
# THREE physical families (not one buggy term) already 44-150 sigma out at bias=0: r_gyro
# rms~144-151, r_vel rms~67-86, r_pos rms~44-84. With loss="huber", f_scale=1.0, residuals this
# far out sit deep in huber's LINEAR regime, where its gradient saturates to a near-constant
# magnitude -- TRF gets almost no curvature/direction signal from any of them, which is why the
# solve stalled instead of slowly converging. Scaling every family's sigma (gyro_sigma,
# accel_sigma, bias_walk_sigma_g/a, and anchor_sigma_g/a which derives from bias_walk_sigma_*)
# by the SAME constant is a pure rescale of the whole residual vector -- it does not change the
# relative weighting between families or the location of the unconstrained optimum, only where
# huber's quadratic/linear boundary falls relative to the actual data. 100x brings the worst
# offender (gyro, rms~150) down to ~1.5 sigma -- just past the threshold, where huber still
# does its job on genuine outlier gaps -- while accel_vel/accel_pos (rms~67-86) land at
# ~0.7-0.9 sigma, back in the quadratic region where gradients are informative. This is a
# diagnostic loosening to let the solver move at all, not a claim that 100x is the "correct"
# noise model -- see module docstring's own caveat that this weighting is a rough engineering
# scale to begin with.
_SIGMA_SCALE = 100.0

# Same transform load_and_calibrate_controller_imu now applies too (as of 2026-09-02),
# superseding its old per-sensor _Y_FLIP @ Rt.R^(±1) -- see module docstring's rot_delta
# section and README findings 7-8 for why this script still can't just call that function
# directly. A precise 180deg flip about X (X unchanged, Y/Z reversed) is physically
# unremarkable: a chip mounted with two axes reversed relative to the device's logical
# convention is a common, deliberate PCB-layout choice, not a bug.
_DIAG_FLIP = np.diag([1.0, -1.0, -1.0])
# rot_delta's anchor sigma: NOT derived from any factory field (the factory JSON has no
# per-axis mounting-rotation uncertainty to reuse, unlike bias_uncertainty) -- an engineering
# judgment call, loose enough that the ~35-72 sigma of residual _DIAG_FLIP alone left behind
# (see README finding 7) can still pull rot_delta to wherever the data wants, tight enough
# that a single global 3-DOF parameter -- already well-constrained by ~15*(n-1) residuals --
# doesn't wander to something unphysical if the solve degenerates.
_ROT_ANCHOR_SIGMA_RAD = np.radians(5.0)


def _seed_velocities(ts_window, world_poses, max_seed_speed=10.0):
    """v_k (3,) per node, nearest-neighbour finite difference (Finding 11):
    for each node, difference against whichever OTHER node in the window is
    closest in time, rather than assuming the adjacent one is always the
    right (or only) choice. In this v1 every node is a good vision frame by
    construction, over a contiguous window, so this reduces to plain
    adjacent differencing -- but written the general way so a later version
    windowing over dropped/rejected vision frames stays correct instead of
    silently degrading. Seeds above max_seed_speed m/s are replaced with
    zero -- "a tracked controller tops out a long way under this even when
    thrown" (Beyley's own comment on the equivalent guard in his code):
    a bad seed can walk the solver's trust region somewhere it never
    recovers from, while zero at least starts in a sane direction."""
    n = len(ts_window)
    v = np.zeros((n, 3))
    for k in range(n):
        candidates = [j for j in (k - 1, k + 1) if 0 <= j < n]
        if not candidates:
            continue
        other = min(candidates, key=lambda j: abs(ts_window[j] - ts_window[k]))
        dt = (ts_window[other] - ts_window[k]) / 1e9
        if dt == 0:
            continue
        v_est = (world_poses[ts_window[other]].t - world_poses[ts_window[k]].t) / dt
        if np.linalg.norm(v_est) > max_seed_speed:
            continue
        v[k] = v_est
    return v


def _build_residual_fn(ts_window, world_poses, t_gyro, gyro_raw_corr, t_accel, accel_raw_corr, g_world,
                        gyro_sigma, accel_sigma, bias_walk_sigma_g, bias_walk_sigma_a,
                        anchor_sigma_g, anchor_sigma_a, lever_arm, rot_anchor_sigma):
    """Returns (residual_fn(x) -> flat np.ndarray, unpack(x) -> (v, b_a, b_g, rot_delta), n).
    n = len(ts_window). x packs [v_0..v_n-1, b_a_0..b_a_n-1, b_g_0..b_g_n-1, rot_delta].

    gyro_raw_corr/accel_raw_corr: mix+bias corrected (ControllerImuAxisCalib.correct) but NOT
    axis-transformed -- unlike the pre-2026-09-02 version of this function, which took
    already-body-frame gyro_body/accel_body. The sensor->body axis transform now depends on
    rot_delta (part of x), so it's recomputed from these raw-corrected streams on every call
    instead of being fixed before the solve -- see module docstring's rot_delta section.

    lever_arm (3,): the accelerometer's FIXED body-frame offset from the body
    origin vision tracks -- a known hardware constant (factory-JSON-derived,
    confirmed 2026-09-02, see visualization/controller_calibration_for_basalt/
    README.md finding 5), not a per-node unknown for this solve to estimate.
    Forwarded to accel_preint_residual's lever-arm correction; bias and rot_delta
    remain the only things x actually solves for, per that finding's recommendation
    not to let a sliding-window bias estimator also re-estimate per-node extrinsics
    (rot_delta is GLOBAL, one value for the whole window, not per-node)."""
    n = len(ts_window)

    def unpack(x):
        v = x[0:3 * n].reshape(n, 3)
        b_a = x[3 * n:6 * n].reshape(n, 3)
        b_g = x[6 * n:9 * n].reshape(n, 3)
        rot_delta = x[9 * n:9 * n + 3]
        return v, b_a, b_g, rot_delta

    def residual_fn(x):
        v, b_a, b_g, rot_delta = unpack(x)
        R_free = Rotation.from_rotvec(rot_delta).as_matrix() @ _DIAG_FLIP
        gyro_body = (R_free @ gyro_raw_corr.T).T
        accel_body = (R_free @ accel_raw_corr.T).T

        parts = []
        for k in range(n - 1):
            ts0, ts1 = ts_window[k], ts_window[k + 1]
            dt = (ts1 - ts0) / 1e9
            if dt <= 0 or dt > _MAX_PAIR_DT_S:
                continue
            R0, p0 = world_poses[ts0].R, world_poses[ts0].t
            R1, p1 = world_poses[ts1].R, world_poses[ts1].t

            r_gyro, _dt_g, r_bias_g = gyro_preint_residual(
                t_gyro, gyro_body, ts0, ts1, R0, R1,
                bias0=b_g[k], bias1=b_g[k + 1], bias_random_walk_std=bias_walk_sigma_g)
            if r_gyro is not None:
                parts.append(r_gyro / (gyro_sigma * np.sqrt(dt)))
                parts.append(r_bias_g)

            r_vel, r_pos, _dt_a, r_bias_a = accel_preint_residual(
                t_accel, accel_body, ts0, ts1, R0, R1, p0, p1, v[k], v[k + 1], g_world,
                bias0=b_a[k], bias1=b_a[k + 1], bias_random_walk_std=bias_walk_sigma_a,
                t_gyro=t_gyro, gyro_body=gyro_body, r=lever_arm)
            if r_vel is not None:
                parts.append(r_vel / (accel_sigma * np.sqrt(dt)))
                parts.append(r_pos / (accel_sigma * dt * np.sqrt(dt)))
                parts.append(r_bias_a)

        # Anchor: node 0's bias pinned toward zero (this project's convention -- bias here
        # means "correction on top of the already-applied factory T=0 calibration", so zero
        # is the honest prior, not a placeholder). Without this the random-walk chain above
        # has a gauge freedom -- see module docstring. rot_delta anchored toward zero (i.e.
        # exactly _DIAG_FLIP) the same way, but loosely -- see _ROT_ANCHOR_SIGMA_RAD.
        parts.append(b_a[0] / anchor_sigma_a)
        parts.append(b_g[0] / anchor_sigma_g)
        parts.append(rot_delta / rot_anchor_sigma)

        return np.concatenate(parts)

    return residual_fn, unpack, n


def _numeric_sparsity(residual_fn, x0: np.ndarray, r0: np.ndarray, eps: float = 1e-6):
    """Boolean sparse (m, n) Jacobian STRUCTURE for residual_fn at x0, found by
    perturbing each variable in turn and recording which residual entries move --
    not hand-derived from the residual layout (which risks silently drifting out
    of sync with the actual code if either changes), self-verifying by
    construction instead.

    Why this matters: each gap's residuals only depend on that gap's own two
    nodes' v/b_a/b_g (see _build_residual_fn) -- structurally block-tridiagonal
    -- but scipy's DEFAULT dense finite-difference Jacobian doesn't know that,
    so it perturbs all 9*n variables independently every solver iteration
    (measured directly during Step 4 implementation: for a 30-node/270-variable
    window, that's ~271 residual_fn calls per iteration, ~5s at this problem's
    ~19ms/call, times however many iterations TRF needs -- impractical even at
    "a few hundred frames" scale, let alone the full recording). Once
    jac_sparsity is known, scipy's sparse-aware differencing groups multiple
    non-interacting columns into a single perturbation via graph colouring, so
    each iteration costs roughly the sparsity pattern's bandwidth instead of the
    full variable count -- this one-time detection pass costs the same as a
    SINGLE dense Jacobian (len(x0)+1 calls) and pays for itself many times over
    across the solve's remaining iterations."""
    m, n = len(r0), len(x0)
    rows, cols = [], []
    for j in range(n):
        xp = x0.copy()
        xp[j] += eps
        rj = residual_fn(xp)
        changed = np.flatnonzero(np.abs(rj - r0) > 1e-12)
        rows.extend(changed.tolist())
        cols.extend([j] * len(changed))
    return coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(m, n))


def plot_bias_trajectories(ctrl_name, ts_window, b_a, b_g, out_path: Path):
    t_s = (np.array(ts_window) - ts_window[0]) / 1e9
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i in range(3):
        axes[0].plot(t_s, b_a[:, i], label=f"b_a[{_AXIS_NAMES[i]}]", linewidth=0.9)
        axes[1].plot(t_s, b_g[:, i], label=f"b_g[{_AXIS_NAMES[i]}]", linewidth=0.9)
    axes[0].set_ylabel("accel bias correction (m/s^2)")
    axes[1].set_ylabel("gyro bias correction (rad/s)")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    fig.suptitle(f"{ctrl_name}: Step 4 estimated bias trajectories (n={len(ts_window)} nodes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("visualization/step4")
    window_size = int(sys.argv[3]) if len(sys.argv) > 3 else _DEFAULT_WINDOW
    max_nfev = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml_config(config_path)

    pose_csv_path = config.get("debug", {}).get("pose_csv")
    if not pose_csv_path or not Path(pose_csv_path).exists():
        raise SystemExit(f"debug.pose_csv not set or missing ({pose_csv_path}) -- run main.py first "
                          f"with debug.pose_csv set")

    poses, _errors = load_pose_csv(pose_csv_path)
    headset_mocap = load_device_mocap("headset")
    mav0_root = Path(config["data"]["root"])

    for ctrl_name in ("left_controller", "right_controller"):
        if ctrl_name not in poses or ctrl_name not in _IMU_FILES:
            continue
        imu_rel_path, lag_ns = _IMU_FILES[ctrl_name]
        imu_path = mav0_root / imu_rel_path
        if not imu_path.exists():
            print(f"[{ctrl_name}] IMU file not found ({imu_path}) -- skipping")
            continue

        ctrl_cfg = config["controllers"][ctrl_name]
        ctrl_json_cfg = load_json_config(ctrl_cfg["config_path"])
        imu_calib = create_imu_calib_from_config(ctrl_json_cfg)

        # Mix+bias corrected but NOT axis-transformed (unlike load_and_calibrate_controller_imu,
        # which now bakes in _DIAG_FLIP -- see module docstring's rot_delta section for why
        # that's still not called directly here: the axis transform is solved for, via
        # rot_delta, as a correction on top of _DIAG_FLIP).
        t_raw, gyro_raw, accel_raw = load_imu_csv(imu_path)
        t_raw = t_raw + lag_ns
        gyro_raw_corr = imu_calib.gyro.correct(gyro_raw.astype(np.float64))
        accel_raw_corr = imu_calib.accel.correct(accel_raw.astype(np.float64))
        t_gyro = t_accel = t_raw

        # Lever arm: FIXED hardware constant, derived directly from the factory JSON's own
        # accel/gyro Rt entries (not re-estimated by this solve -- see finding 5 in
        # visualization/controller_calibration_for_basalt/README.md, confirmed 2026-09-02
        # three independent ways and cross-validated against mocap2gt). accel.T_rt.compose(
        # gyro.T_rt.inverse()) is accel.Rt ∘ gyro.Rt^-1 -- rotation is near-identity
        # (sub-0.2°, noise), translation is the ~85mm lever arm that Step 4's original
        # lever-arm-free residuals couldn't explain no matter how far bias was allowed to move.
        lever_arm = imu_calib.accel.T_rt.compose(imu_calib.gyro.T_rt.inverse()).t
        print(f"[{ctrl_name}] lever arm (m): {lever_arm}  |.|={np.linalg.norm(lever_arm) * 1000:.2f}mm")

        world_poses = world_vision_poses(poses[ctrl_name], headset_mocap)
        ts_sorted = sorted(world_poses.keys())
        if len(ts_sorted) < 10:
            print(f"[{ctrl_name}] not enough world-frame vision poses ({len(ts_sorted)}) -- skipping")
            continue

        # g_world bootstrap uses the x0 transform (rot_delta=0, i.e. exactly _DIAG_FLIP) for
        # consistency with what residual_fn(x0) will itself compute -- see module docstring.
        gyro_body_x0 = (_DIAG_FLIP @ gyro_raw_corr.T).T
        accel_body_x0 = (_DIAG_FLIP @ accel_raw_corr.T).T
        g_world, n_used, n_total = low_motion_bootstrap_g_world(
            t_gyro, gyro_body_x0, t_accel, accel_body_x0, world_poses)
        if g_world is None:
            print(f"[{ctrl_name}] not enough low-motion frames for g_world bootstrap ({n_used}/{n_total}) "
                  f"-- skipping")
            continue

        ts_window = ts_sorted[:window_size]
        n = len(ts_window)
        span_s = (ts_window[-1] - ts_window[0]) / 1e9
        print(f"[{ctrl_name}] window: {n} nodes, {span_s:.2f}s span, g_world={g_world} "
              f"(|g|={np.linalg.norm(g_world):.3f} m/s^2, {n_used}/{n_total} low-motion frames used)")

        gyro_sigma = np.maximum(imu_calib.gyro.noise_std, _MIN_SIGMA) * _SIGMA_SCALE
        accel_sigma = np.maximum(imu_calib.accel.noise_std, _MIN_SIGMA) * _SIGMA_SCALE
        bias_walk_sigma_g = np.maximum(imu_calib.gyro.bias_uncertainty, _MIN_SIGMA) * _SIGMA_SCALE
        bias_walk_sigma_a = np.maximum(imu_calib.accel.bias_uncertainty, _MIN_SIGMA) * _SIGMA_SCALE
        anchor_sigma_g = bias_walk_sigma_g * _ANCHOR_SIGMA_SCALE
        anchor_sigma_a = bias_walk_sigma_a * _ANCHOR_SIGMA_SCALE

        v_seed = _seed_velocities(ts_window, world_poses)
        x0 = np.concatenate([v_seed.ravel(), np.zeros(3 * n), np.zeros(3 * n), np.zeros(3)])

        # Sliced to the window's own time span (see slice_imu_to_window's docstring for
        # why this matters for solve speed, not just memory). Raw-corrected, not yet
        # axis-transformed -- see _build_residual_fn's docstring.
        t_gyro_w, gyro_raw_corr_w = slice_imu_to_window(t_gyro, gyro_raw_corr, ts_window[0], ts_window[-1])
        t_accel_w, accel_raw_corr_w = slice_imu_to_window(t_accel, accel_raw_corr, ts_window[0], ts_window[-1])

        residual_fn, unpack, _ = _build_residual_fn(
            ts_window, world_poses, t_gyro_w, gyro_raw_corr_w, t_accel_w, accel_raw_corr_w, g_world,
            gyro_sigma, accel_sigma, bias_walk_sigma_g, bias_walk_sigma_a,
            anchor_sigma_g, anchor_sigma_a, lever_arm, _ROT_ANCHOR_SIGMA_RAD)

        res0 = residual_fn(x0)
        rms0 = float(np.sqrt(np.mean(res0 ** 2)))

        t_sparsity = time.time()
        sparsity = _numeric_sparsity(residual_fn, x0, res0)
        print(f"[{ctrl_name}] Jacobian sparsity detected in {time.time() - t_sparsity:.1f}s: "
              f"{sparsity.nnz}/{sparsity.shape[0] * sparsity.shape[1]} nonzero "
              f"({100 * sparsity.nnz / (sparsity.shape[0] * sparsity.shape[1]):.2f}%)")

        result = least_squares(residual_fn, x0, method="trf", loss="huber", f_scale=1.0,
                                jac_sparsity=sparsity, max_nfev=max_nfev)

        res1 = residual_fn(result.x)
        rms1 = float(np.sqrt(np.mean(res1 ** 2)))
        v_opt, b_a_opt, b_g_opt, rot_delta_opt = unpack(result.x)
        rot_delta_deg = np.degrees(np.linalg.norm(rot_delta_opt))
        R_free_opt = Rotation.from_rotvec(rot_delta_opt).as_matrix() @ _DIAG_FLIP

        print(f"[{ctrl_name}] solve {'converged' if result.success else 'DID NOT CONVERGE'} "
              f"({result.nfev} evals): residual RMS (in sigma units) {rms0:.3f} -> {rms1:.3f} "
              f"over {len(res0)} residuals")
        print(f"[{ctrl_name}] rot_delta: {np.degrees(rot_delta_opt)} deg (|.|={rot_delta_deg:.3f} deg from "
              f"_DIAG_FLIP)  resulting R:\n{R_free_opt}")
        print(f"[{ctrl_name}] b_a (m/s^2): mean={b_a_opt.mean(axis=0)} std={b_a_opt.std(axis=0)} "
              f"  factory BiasUncertainty={imu_calib.accel.bias_uncertainty}")
        print(f"[{ctrl_name}] b_g (rad/s): mean={b_g_opt.mean(axis=0)} std={b_g_opt.std(axis=0)} "
              f"  factory BiasUncertainty={imu_calib.gyro.bias_uncertainty}")

        plot_bias_trajectories(ctrl_name, ts_window, b_a_opt, b_g_opt,
                                out_dir / f"bias_estimation_{ctrl_name}_trajectories.png")
        print(f"[{ctrl_name}] saved plot to {out_dir}/bias_estimation_{ctrl_name}_trajectories.png\n")


if __name__ == "__main__":
    main()
