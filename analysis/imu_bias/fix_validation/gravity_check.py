"""gravity_check.py -- quiet-window accelerometer consistency with gravity, on the recorded stream.
Quiet window (0.25 s hop, 0.25 s length): mocap IMU-point speed < 0.05 m/s, speed CHANGE between window halves < 0.02 m/s
(=> mean linear accel < 0.08 m/s^2), mocap angular rate < 0.3 rad/s, >= 40 IMU samples.  In a quiet window the mean specific force is
f = R^T (0, g, 0) (additive convention, mocap room y is gravity-aligned to ~0.06 deg, see imu_data.MOCAP_ROOM_G_WORLD) plus bias/gain.
Model in the CSV (sensor) axes:  f_i = (1 + k_i) * g0 * u_i + b_i,  u = R_world_imu^T yhat, g0 = 9.80665.
Reports per recording/controller: n quiet windows/seconds, |mean f| of every stream variant, per-axis gain k and bias b (LS), isotropic
gain k_iso, and the decomposition of the |mean f| excess into gain vs bias."""
import numpy as np
import pandas as pd
import fv_common as C

G0 = 9.80665
YHAT = np.array([0.0, 1.0, 0.0])
W = 0.25e9
rows = []
qwin = {}
for ctrl in C.CTRLS:
    for name in C.MODERATE:
        rd = C.rec_dir(name); dev = C.load_mocap_device(rd, ctrl); mo = C.MocapPose(dev)
        t, g_csv, a_csv = C.raw_csv(rd, ctrl)
        starts = np.arange(t[0] + 0.5e9, t[-1] - W - 0.5e9, 0.25e9).astype(np.int64)
        U, F, wsel = [], [], []
        for s in starts:
            q = np.array([s, s + int(W / 2), s + int(W)], dtype=np.int64)
            if not mo.valid(q).all():
                continue
            tq = (s + np.linspace(0, W, 19)).astype(np.int64)
            if not mo.valid(tq).all():
                continue
            pq = mo.pos_imu(tq); tt = (tq - s) / 1e9
            v1 = np.array([np.polyfit(tt[:10], pq[:10, i], 1)[0] for i in range(3)]); v2 = np.array([np.polyfit(tt[9:], pq[9:, i], 1)[0] for i in range(3)])
            vm = 0.5 * (v1 + v2)
            R0, R1 = mo.R_world_imu(q[[0]]), mo.R_world_imu(q[[2]])
            om = np.linalg.norm((R0[0].inv() * R1[0]).as_rotvec()) / (W / 1e9)
            if np.linalg.norm(vm) > 0.05 or np.linalg.norm(v2 - v1) > 0.02 or om > 0.3:
                continue
            sel = (t >= s) & (t <= s + W)
            if sel.sum() < 40:
                continue
            Rc = mo.R_world_imu(np.array([s + int(W / 2)]))[0].as_matrix()
            U.append(Rc.T @ YHAT); F.append(a_csv[sel].mean(axis=0)); wsel.append(s)
        U, F = np.array(U), np.array(F)
        qwin[(name, ctrl)] = (U, F)
        if len(U) < 3:
            rows.append(dict(rec=name, ctrl=ctrl, n_win=len(U))); continue
        # per-axis LS: f_i = (1+k_i) g0 u_i + b_i
        ks, bs, ident = [], [], []
        for i in range(3):
            A = np.stack([G0 * U[:, i], np.ones(len(U))], 1)
            coef, *_ = np.linalg.lstsq(A, F[:, i], rcond=None)
            ks.append(coef[0] - 1); bs.append(coef[1]); ident.append(U[:, i].std())
        # isotropic gain + per-axis bias (joint): f_i = (1+k) g0 u_i + b_i
        A = np.zeros((3 * len(U), 4)); y = F.ravel()
        for j in range(len(U)):
            for i in range(3):
                A[3 * j + i, 0] = G0 * U[j, i]; A[3 * j + i, 1 + i] = 1.0
        coef, *_ = np.linalg.lstsq(A, y, rcond=None); k_iso = coef[0] - 1; b_iso = coef[1:]
        mag = lambda Fm: np.linalg.norm(Fm.mean(axis=0))
        # magnitude of the MEAN specific force over all quiet windows, per stream variant (body frame magnitude = axes-independent)
        c = C.factory_calib(ctrl, 1)
        a_old = (c.accel.mix0 @ F.T).T + c.accel.bias0        # note: applied to window means (affine -> commutes with the mean)
        row = dict(rec=name, ctrl=ctrl, n_win=len(U), quiet_s=len(U) * 0.25,
                   mean_absf_csv=float(np.mean(np.linalg.norm(F, axis=1))),
                   absf_L0_old=float(np.mean(np.linalg.norm(a_old, axis=1))),
                   absf_L1_drop2nd=float(np.mean(np.linalg.norm(F, axis=1))),
                   absf_L2_scale_only=float(C.SCALE * np.mean(np.linalg.norm(a_old, axis=1))),
                   absf_L3_new=float(C.SCALE * np.mean(np.linalg.norm(F, axis=1))),
                   k_x=ks[0], k_y=ks[1], k_z=ks[2], b_x=bs[0], b_y=bs[1], b_z=bs[2], k_iso=k_iso, b_iso_norm=float(np.linalg.norm(b_iso)),
                   u_std_x=ident[0], u_std_y=ident[1], u_std_z=ident[2])
        # residual after removing per-axis fit
        res = []
        for i in range(3):
            res.append(F[:, i] - ((1 + ks[i]) * G0 * U[:, i] + bs[i]))
        row["fit_rms"] = float(np.sqrt(np.mean(np.square(res))))
        # projection of the fitted bias on the gravity direction (mean over windows) -> excess of |f| due to bias
        row["bias_along_gravity"] = float(np.mean([np.dot(np.array(bs), u) for u in U]))
        rows.append(row)
        print(f"{name:13s} {ctrl:16s} windows {len(U):3d} ({len(U)*0.25:5.1f} s)  |f| csv {row['mean_absf_csv']:.3f}  L3 {row['absf_L3_new']:.3f}  k_iso {100*k_iso:+.2f}%  per-axis k {100*np.array(ks).round(4)}  b {np.round(bs,3)}", flush=True)
df = pd.DataFrame(rows); df.to_csv(C.HERE / "results" / "gravity_check.csv", index=False)
ok = df.dropna(subset=["mean_absf_csv"])
print("\nrecordings x controllers with quiet windows:", len(ok), "of", len(df))
for col in ["absf_L0_old", "absf_L1_drop2nd", "absf_L2_scale_only", "absf_L3_new"]:
    print(f"  mean |f| quiet  {col:20s}  min {ok[col].min():.3f}  max {ok[col].max():.3f}  median {ok[col].median():.3f}   (g0 = {G0})")
print("  isotropic gain k_iso on the CSV: min %.4f max %.4f median %.4f  (driver constant predicts +0.0197)" % (ok.k_iso.min(), ok.k_iso.max(), ok.k_iso.median()))
print("  per-axis k (CSV axes) median: x %.4f y %.4f z %.4f" % (ok.k_x.median(), ok.k_y.median(), ok.k_z.median()))
print("  bias along gravity (m/s^2) median %.3f, range [%.3f, %.3f]" % (ok.bias_along_gravity.median(), ok.bias_along_gravity.min(), ok.bias_along_gravity.max()))
