"""Known-answer tests for the SIMULATOR (run before any estimator exists).
Each test states its expected value; project functions are used as the independent reference."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from simlib import *
from src.imu_data import integrate_gyro_segment, integrate_accel_to_position
from src.mocap_data import DeviceMocap, world_pose, load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker

results = []
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail)); print(("PASS " if ok else "FAIL ") + name + "  " + detail)

# ---------- T1: Truth kinematics self-consistency (R integrates omega; spline derivatives)
d0 = Scenario(noise_g=0, noise_a=0, outlier_frac=0, sig_rot_deg=0, sig_pos_mm=0)
sim = simulate("static_dark", "right", d0, t_max_s=40)
tr = sim.truth
tt = np.linspace(tr.t0 + 5, tr.t0 + 30, 400)
h = 1e-4
Rp, Rm = tr.R(tt + h), tr.R(tt - h)
dRdt = (Rp - Rm) / (2 * h)
R = tr.R(tt); w = tr.w(tt)
Rdot_model = np.einsum("nij,njk->nik", R, np.array([skew(x) for x in w]))
err = np.abs(dRdt - Rdot_model).max()
wmax = np.linalg.norm(w,axis=1).max()
check("T1a dR/dt = R[omega]x (max abs err, relative to |omega|max)", err/wmax < 3e-3, f"{err:.2e} abs, {err/wmax:.1e} rel (|omega| max {wmax:.2f} rad/s; slerp between 1 kHz nodes)")
vfd = (tr.p(tt + h) - tr.p(tt - h)) / (2 * h)
check("T1b spline v = dp/dt", np.abs(vfd - tr.v(tt)).max() < 1e-6, f"{np.abs(vfd - tr.v(tt)).max():.1e}")
afd = (tr.v(tt + h) - tr.v(tt - h)) / (2 * h)
check("T1c spline a = dv/dt", np.abs(afd - tr.a(tt)).max() < 1e-3, f"{np.abs(afd - tr.a(tt)).max():.1e}")

# ---------- T2: synthetic gyro (no noise/bias) integrated by the PROJECT's integrate_gyro_segment reproduces truth R
t_ns, gy, ac = sim.t_imu_ns, sim.gyro, sim.accel
ts = sim.t_imu_s
errs = []
for a in np.linspace(3, 30, 40):
    i0 = np.searchsorted(ts, a); i1 = np.searchsorted(ts, a + 0.5)
    dR = integrate_gyro_segment(t_ns, gy, t_ns[i0], t_ns[i1])
    R0, R1 = tr.R(ts[i0])[0], tr.R(ts[i1])[0]
    errs.append(np.degrees(np.linalg.norm(Log(dR.T @ (R0.T @ R1)))))
check("T2 project gyro integration vs sim truth rotation over 0.5 s (max deg)", max(errs) < 0.05,
      f"max {max(errs):.4f} deg, median {np.median(errs):.4f}")

# ---------- T3: synthetic accel (no noise/bias, with lever arm) double-integrated by project reproduces truth position
perr = []
for a in np.linspace(3, 30, 40):
    i0 = np.searchsorted(ts, a); i1 = np.searchsorted(ts, a + 0.04)
    R0, R1 = tr.R(ts[i0])[0], tr.R(ts[i1])[0]
    v0 = tr.v(ts[i0])[0]
    dp = integrate_accel_to_position(t_ns, ac.astype(float), t_ns[i0], t_ns[i1], R0, R1, v0, G_WORLD,
                                     t_gyro=t_ns, gyro_body=gy.astype(float), r=LEVER_ARM["right"])
    dp_true = tr.p(ts[i1])[0] - tr.p(ts[i0])[0]
    perr.append(np.linalg.norm(dp - dp_true) * 1000)
check("T3 project accel dead-reckoning (lever arm, gravity) vs sim truth over 0.04 s = one vision gap, the integrator's documented horizon (max mm)", max(perr) < 0.5,
      f"max {max(perr):.3f} mm, median {np.median(perr):.3f}")
# T3b: WRONG lever-arm sign must be detectable (guards against the test being insensitive)
perr2 = []
for a in np.linspace(3, 30, 40):
    i0 = np.searchsorted(ts, a); i1 = np.searchsorted(ts, a + 0.04)
    R0, R1 = tr.R(ts[i0])[0], tr.R(ts[i1])[0]
    dp = integrate_accel_to_position(t_ns, ac.astype(float), t_ns[i0], t_ns[i1], R0, R1, tr.v(ts[i0])[0], G_WORLD,
                                     t_gyro=t_ns, gyro_body=gy.astype(float), r=None)
    perr2.append(np.linalg.norm(dp - (tr.p(ts[i1])[0] - tr.p(ts[i0])[0])) * 1000)
check("T3b test is sensitive: ignoring the lever arm gives larger error", np.median(perr2) > 3 * np.median(perr),
      f"median with lever {np.median(perr):.3f} mm vs without {np.median(perr2):.3f} mm")

# ---------- T4: my vectorised marker->IMU pose chain equals src.mocap_data.world_pose
root = REC_ROOT / RECORDINGS["static_dark"]
dv = root / "mocap_filtered" / "ctrlright"
t, pos, quat = load_mocap_csv(dv / "data.csv")
fine = load_mocap_fine_offset_ns(dv / "drift_check/1chunk/drift_check.json")
Tim = load_T_imu_marker(CALIB_DIR / "controller_right_calib.json")
dev = DeviceMocap(t, pos, quat, 0.0, Tim)
mx = []
for q in np.linspace(t[100], t[-100], 25).astype(np.int64):
    T = world_pose(dev, int(q))
    j = np.searchsorted(t, q, "right") - 1
    Rm = Rot.from_quat(quat[j].astype(float)).as_matrix()
    Rwi = Rm @ Tim.R.T
    # compare at the exact sample j (no interpolation): query at t[j]
    Tj = world_pose(dev, int(t[j]))
    if Tj is None: continue
    pj = pos[j].astype(float) - Rwi @ Tim.t
    mx.append(max(np.abs(Tj.R - Rwi).max(), np.abs(Tj.t - pj).max()))
check("T4 vectorised marker->IMU pose chain == src.mocap_data.world_pose", max(mx) < 1e-5, f"max diff {max(mx):.1e}")

# ---------- T5: bias injection sign: gyro = omega + b  =>  project integration with (gyro - b) recovers truth
b = np.array([0.02, -0.03, 0.01])
sb = simulate("static_dark", "right", Scenario(noise_g=0, noise_a=0, outlier_frac=0, sig_rot_deg=0, sig_pos_mm=0, b_g0=b), t_max_s=20)
ts2 = sb.t_imu_s; e0, e1 = [], []
for a in np.linspace(2, 15, 20):
    i0 = np.searchsorted(ts2, a); i1 = np.searchsorted(ts2, a + 0.5)
    R_true = sb.truth.R(ts2[i0])[0].T @ sb.truth.R(ts2[i1])[0]
    dR_c = integrate_gyro_segment(sb.t_imu_ns, sb.gyro - b, sb.t_imu_ns[i0], sb.t_imu_ns[i1])
    dR_n = integrate_gyro_segment(sb.t_imu_ns, sb.gyro, sb.t_imu_ns[i0], sb.t_imu_ns[i1])
    e0.append(np.degrees(np.linalg.norm(Log(dR_c.T @ R_true)))); e1.append(np.degrees(np.linalg.norm(Log(dR_n.T @ R_true))))
check("T5 subtracting the true bias recovers truth; not subtracting leaves ~|b|*T", np.max(e0) < 0.05 and np.median(e1) > 0.3,
      f"corrected max {np.max(e0):.3f} deg; uncorrected median {np.median(e1):.3f} deg (expected ~{np.degrees(np.linalg.norm(b)*0.5):.2f})")

# ---------- T6: noise levels reproduce their nominal sigma (statistical)
sn = simulate("static_dark", "right", Scenario(), t_max_s=30)
sq = simulate("static_dark", "right", Scenario(noise_g=0, noise_a=0), t_max_s=30)
ng = (sn.gyro - sq.gyro).std(); na = (sn.accel - sq.accel).std()
check("T6 IMU noise sigma reproduced", abs(ng - 7e-4) < 3e-5 and abs(na - 6.5e-3) < 3e-4, f"gyro {ng:.2e} accel {na:.2e}")

n_fail = sum(1 for r in results if not r[1])
print(f"\n{len(results)-n_fail}/{len(results)} passed")
import json
json.dump([dict(name=n, ok=o, detail=dt) for n, o, dt in results], open(Path(__file__).parent / "test_sim_results.json", "w"), indent=1)
sys.exit(1 if n_fail else 0)
