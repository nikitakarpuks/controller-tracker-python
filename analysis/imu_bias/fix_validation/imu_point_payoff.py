"""imu_point_payoff.py -- mocap-truth payoff of the loader variants, in the CSV/sensor frame (no lever arm involved):
  gyro : rotation-prediction error over gap T  = angle(Rm^T Rg), Rm = mocap relative rotation of the IMU frame, Rg = integrated gyro
  accel: position dead-reckoning error of the ACCELEROMETER POINT over gap T from a mocap initial state (p0, v0 from a +-40 ms linear
         fit of the mocap IMU-point position) with mocap orientation.
Same start sampling and metrics as the factory audit (fa_payoff.py) so it doubles as a SANITY GATE: OLD loader ~196 mm accel at 1 s and
~1.70 deg gyro at 1 s (pooled 6 moderate recordings); NEW ~75 mm and ~1.35 deg.  No fitting -> no leakage.
Writes results/imu_point_gyro.csv, results/imu_point_accel.csv (one row per start sample, all variants) + prints tables with paired CIs."""
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import fv_common as C

NST = 300
GT = [0.044, 0.1, 0.3, 1.0]
AT = [0.1, 0.3, 1.0]
G0 = C.G_ABS


def variants_csv(ctrl, g_csv, a_csv):
    c = C.factory_calib(ctrl, 1)
    g_old = (c.gyro.mix0 @ g_csv.T).T + c.gyro.bias0
    a_old = (c.accel.mix0 @ a_csv.T).T + c.accel.bias0
    gyro = {"L0_OLD": g_old, "L1_drop2nd": g_csv}
    accel = {"L0_OLD": a_old, "L1_drop2nd": a_csv, "L2_scale_only": C.SCALE * a_old, "L3_NEW": C.SCALE * a_csv}
    return gyro, accel


rows_g, rows_a = [], []
for ctrl in C.CTRLS:
    for j, name in enumerate(C.MODERATE):
        rd = C.rec_dir(name)
        dev = C.load_mocap_device(rd, ctrl); mo = C.MocapPose(dev)
        t, g_csv, a_csv = C.raw_csv(rd, ctrl)
        gyro, accel = variants_csv(ctrl, g_csv, a_csv)
        rng = np.random.default_rng(100 + j)
        for T in GT:
            dtn = int(T * 1e9)
            cand = np.arange(t[0] + 0.5e9, t[-1] - dtn - 0.5e9, 1e7).astype(np.int64)
            cand = cand[mo.valid(cand) & mo.valid(cand + dtn)]
            starts = rng.choice(cand, min(NST, len(cand)), replace=False)
            R0 = mo.R_world_imu(starts); R1 = mo.R_world_imu(starts + dtn)
            for i, s0 in enumerate(starts):
                Rm = (R0[i].inv() * R1[i]).as_matrix()
                row = dict(rec=name, ctrl=ctrl, T=T, t_start=int(s0))
                for v, gs in gyro.items():
                    Rg = C.IH.integrate_gyro_segment(t, gs, int(s0), int(s0) + dtn)
                    row[v] = float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rm.T @ Rg).as_rotvec())))
                rows_g.append(row)
        rng = np.random.default_rng(200 + j)
        for T in AT:
            dtn = int(T * 1e9)
            cand = np.arange(t[0] + 0.5e9, t[-1] - dtn - 0.5e9, 2e7).astype(np.int64)
            cand = cand[mo.valid(cand) & mo.valid(cand + dtn) & mo.valid(cand - int(0.05e9)) & mo.valid(cand + int(0.05e9))]
            starts = rng.choice(cand, min(NST, len(cand)), replace=False)
            for s0 in starts:
                tq = (s0 + np.linspace(-0.04e9, 0.04e9, 17)).astype(np.int64); pq = mo.pos_imu(tq); tt = (tq - s0) / 1e9
                v0 = np.array([np.polyfit(tt, pq[:, i], 1)[0] for i in range(3)]); p0 = np.array([np.polyval(np.polyfit(tt, pq[:, i], 1), 0) for i in range(3)])
                sel = np.flatnonzero((t >= s0) & (t <= s0 + dtn))
                tI = np.concatenate(([s0], t[sel], [s0 + dtn])).astype(np.int64)
                RI = mo.R_world_imu(tI).as_matrix(); ts = (tI - tI[0]) / 1e9
                p_true = mo.pos_imu(np.array([s0 + dtn]))[0]
                row = dict(rec=name, ctrl=ctrl, T=T, t_start=int(s0))
                for v, fs in accel.items():
                    fI = np.stack([np.interp(tI, t, fs[:, i]) for i in range(3)], 1)
                    a_w = np.einsum("nij,nj->ni", RI, fI) + G0
                    p = p0 + v0 * ts[-1] + C.cum2(ts, a_w)[-1]
                    row[v] = float(np.linalg.norm(p - p_true) * 1000)
                rows_a.append(row)
    print(ctrl, "done", flush=True)

g = pd.DataFrame(rows_g); a = pd.DataFrame(rows_a)
g.to_csv(C.HERE / "results" / "imu_point_gyro.csv", index=False); a.to_csv(C.HERE / "results" / "imu_point_accel.csv", index=False)


def report(df, variants, Ts, unit, title, ref="L0_OLD"):
    print(f"\n{title}  (median [p90] {unit}; pooled 6 moderate recordings; paired mean diff vs {ref} with 95% block-bootstrap CI)")
    lines = []
    for ctrl in list(C.CTRLS) + ["both"]:
        s0 = df if ctrl == "both" else df[df.ctrl == ctrl]
        print(f"  {ctrl}")
        print(f"  {'variant':16s}" + "".join(f"{'T=' + str(T):>34s}" for T in Ts))
        for v in variants:
            line = f"  {v:16s}"
            for T in Ts:
                s = s0[s0["T"] == T]
                e = s[v].to_numpy(); line += f"{np.median(e):8.3f} [{np.percentile(e, 90):7.3f}]"
                if v != ref:
                    d = e - s[ref].to_numpy(); lo, hi = C.block_bootstrap_ci(d, s["t_start"].to_numpy(), n=500)
                    line += f" d={d.mean():+7.3f}[{lo:+6.3f},{hi:+6.3f}]"
                else:
                    line += " " * 26
            print(line)


report(g, ["L0_OLD", "L1_drop2nd"], GT, "deg", "GYRO rotation-prediction error vs mocap")
report(a, ["L0_OLD", "L1_drop2nd", "L2_scale_only", "L3_NEW"], AT, "mm", "ACCEL position dead-reckoning error (accelerometer point, mocap init) vs mocap")
