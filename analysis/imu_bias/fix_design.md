# !! REVIEW OUTCOME (2026-09-23) -- Change 2 and test T3 below are SUPERSEDED !!

An independent adversarial review plus my own numerical check found the lever-arm derivation below has the
bridge convention BACKWARDS. The bridge file header ("p_mocapAccelImu = R p_ledRef + t") documents it wrongly;
the project's own residual (compare_vision_mocap.py:165, T_vision.compose(bridge) ~ mocap_rel) and
mocap_data.load_mocap_bridge make the bridge map IMU-frame points to LED-frame points, so the accelerometer
origin in the LED frame is bridge.t (left (5.6, 7.4, -83.9) mm, right (-5.7, 7.6, -82.8) mm), NOT -R^T t (which
flips x and moves y by ~1.6 mm, an ~11 mm error). The real defect is a FRAME bug: the factory translation is right
but was read without Rt's rotation. Shipped fix: lever arm = -R_acc^T t_acc = accel.T_rt.inverse().t (minus the
gyro's, zero) -- src/imu_data.accel_lever_arm_body -- mocap-independent, within 2.6 / 3.9 mm of bridge.t, and
the best of 8 candidate readings on held-out data (fix_validation). No config vector and no bridge-consistency
tolerance are needed. Change 1 (loader) shipped as designed, plus a startup guard (median quiet |accel| band
9.70..10.02). See fix_impl/README.md for what was implemented and tested.

---

# Design: controller IMU loader fix + accel lever-arm fix (for independent review)

Status: DESIGN ONLY. No source edited yet. Evidence: analysis/imu_bias/REPORT.md and the factory_audit /
realdata / oracle folders next to it. Reviewer: please try to break this on conventions, signs, frames,
hidden consumers, and evidence quality; do not just agree.

## Change 1: loader (src/imu_data.py load_and_calibrate_controller_imu)

Current: raw CSV -> ControllerImuAxisCalib.correct(): out = mix0 @ raw + bias0 -> D @ out, with
D = diag(1,-1,-1) (_DIAG_FLIP). Timestamps: + lag_ns.

Evidence the recorded stream is ALREADY factory-corrected (factory_audit): Monado
wmr_controller_hp.c:258-276 (full path:
/home/nikitakarpuks/Cpp_projects/monado-dev-constellation-controller-tracking/src/xrt/drivers/wmr/wmr_controller_hp.c;
conversion function at lines 175-183) applies, per sample:
  acc = counts / (98000/2)         [comment: "1g is approximately 490,000. @todo: Confirm the scale"]
  acc = mix_matrix @ acc ; acc += bias_offsets ; acc = P_oxr_acc.orientation * acc
  gyro the same with vec3_from_wmr_controller_gyro, mix, bias, P_oxr_gyr rotation
and the recorder (wmr_source.c:396-405 / wmr_controller_base.c:493-520 / t_euroc_recorder.cpp:188) writes
that sample. Data agree: "CSV as recorded" beats "project loader" in 12/12 recording-controller pairs;
"CSV is raw" is rejected at 3-5 sigma.

Divisor 98000/2 = 49000 => output 10.000 per 490000 counts. If 1 g = 490000 counts (the driver's own
observation) the output is in units where 1 g = 10.0, so true m/s^2 = output * 9.80665/10.0 = output *
0.980665. Measured accel gain error after the loader: 1.9-2.5 %, collapses to ~0 after x0.980665.

Proposed: new behaviour
  gyro_body  = D @ raw_gyro
  accel_body = 0.980665 * (D @ raw_accel)
  t          = t_raw + lag_ns
with two switches (defaults = the fix): factory_corrected_input=True, accel_scale=ACCEL_DRIVER_SCALE
(=9.80665/10.0). factory_corrected_input=False + accel_scale=1.0 reproduces the legacy chain bit-for-bit.
main.py reads two config keys (imu.recorded_stream_factory_corrected, imu.accel_driver_scale) and passes them.
Offline scripts use the function defaults (same pattern as the shared lag_ns change).
NOT changed: residual per-session gyro/accel bias (~0.01 rad/s, ~0.1-0.16 m/s^2), gyro gain/misalignment K,
entry_index, lag. create_imu_calib_from_config stays (noise, BiasUncertainty, fallback lever arm).

## Change 2: accel lever arm (main.py:199)

Current: lever_arm = accel.T_rt.compose(gyro.T_rt.inverse()).t = (34.6,-77.5,2.8) mm left,
(-33.2,-80.2,2.9) mm right. Consumers: src/imu_data._lever_arm_correction (used by
integrate_accel_segment / integrate_accel_to_position / accel_preint_residual / predict_world_pose /
dead_reckon_dense), which SUBTRACTS  alpha x r + omega x (omega x r)  from the accel reading, with r = the
accelerometer's position relative to the body origin that vision tracks, expressed in the body frame
(gyro_body/accel_body frame = D @ sensor frame).

Rigid-body kinematics: point at r has acceleration a_pt = a_origin + alpha x r + omega x (omega x r);
specific force f_body = R^T (a_pt - g) (plus bias etc.), so a_origin-term = f - correction. Consistent.

Proposed r: from the mocap bridge T_ledRef_mocapAccelImu (src/mocap_data.load_mocap_bridge; Transform
convention p_accelImu = R @ p_led + t, Transform.compose(a,b): R_a R_b, R_a t_b + t_a). The accelerometer
origin (p_accelImu = 0) in the LED frame is p_led = -R^T t. Values (current shipped bridges):
left (-5.3, 5.8, -84.0) mm, right (5.5, 5.6, -83.0) mm; magnitudes 84.4 / 83.4 mm match the factory
85.0 / 86.9 mm, but the directions are 97 deg apart (verified by direct computation). An independent
regression on held-out static_dark data (realdata investigator) agrees with the bridge-derived vector to
10-17 mm/axis, and the mocap IMU position was fitted to be the accelerometer position (extra lever arm
0.5-7 mm, oracle). Frame subtlety: LED frame vs body frame (D @ sensor) differ by a small fitted rotation
(~1 deg) => <= 1.5 mm at 85 mm; ignored. Held-out payoff: accel position error at 1 s 0.42 -> 0.23 m.

Proposed plumbing: config key controllers.<ctrl>.imu_accel_lever_arm_m: [x, y, z] (null => legacy factory
derivation); helper controller_accel_lever_arm(ctrl_key, config, factory_fallback) in src/mocap_data.py;
main.py and the 6 offline scripts that rebuild the same expression call it. A unit test asserts the shipped
config value equals -R^T t of the shipped bridge within 2 mm (so a future bridge refit cannot silently
diverge without the test failing).

## Test plan (exact-value, per project rule)

T1 loader synthetic: raw gyro (0.1,0.2,0.3), accel (1,2,3) -> gyro (0.1,-0.2,-0.3), accel
0.980665*(1,-2,-3) exactly; legacy switches reproduce mix0@raw+bias0 then D; t + lag exact.
T2 loader on a real CSV slice: new default == D@raw (gyro) and 0.980665*D@raw (accel) bit-exactly; and equals
the factory-audit "CSV x 9.80665/10" candidate stream.
T3 lever helper: config value returned as float64 (3,); null falls back to exact factory composition;
shipped values within 2 mm of -R^T t of shipped bridges; magnitude 80-90 mm; left/right sign pattern.
T4 physics "prove the bug, prove the fix": body origin at rest (a_origin=0), rotating at constant omega about
a fixed body axis (R(t) = exp(t [omega]x)), accel synthesised as f = R^T(-g) + omega x (omega x r_true)
(+ alpha x r_true); integrating with r_true gives position error ~0 over 0.5 s; integrating with the legacy
factory vector (different direction) gives error at least X; also r=0 gives a third, different error. Uses the
project's own integrate_accel_segment / integrate_accel_to_position.
T5 no-op/invariance: with omega=0 the lever arm has no effect (any r gives identical result).
T6 real-data regression (script, not unit test): accel dead-reckoning gap error on static_dark and walk_medium
before/after (old loader+old lever vs new), expected roughly -50 % at 0.3 s and -60 % at 1 s; gyro rotation error
at 1 s 1.70 -> 1.35 deg. Plus an end-to-end main.py subsequence run (no crash, startup log states the new
settings, fused-vs-mocap error not worse).

## Known risks to review

R1 Behaviour change for anything not recorded through this Monado build (default assumes corrected input).
R2 Tuned constants downstream were fit on the old stream (coast_trust_*, accel/gyro jump thresholds):
   expected second-order (accel -1.9 % scale, ~0.1-0.2 m/s^2 offset shift), needs a live-run check.
R3 LiveGravityEstimator (rig-frame g estimated from accel) self-corrects; MOCAP_ROOM_G_WORLD is fixed at
   9.81: now consistent with the scaled accel (before: 10.0x accel vs 9.81 gravity mismatch ~ 0.2-0.3 m/s^2).
R4 All numbers come from one ~25 min session / one controller pair; mocap-referenced fits absorb any
   constant controller T_imu_marker error (that solve did not converge, cost 1.0e6 / 3.1e6).
R5 The bridge is refit occasionally (shifts of 1-3 mm): the consistency test tolerance is 2 mm.
