"""provenance_check.py -- cheap runtime statistics that tell an 'already driver-corrected' recorded IMU stream from a 'raw sensor' one.
  A (axis frame, needs vision): Wahba rotation Q that best maps the recorded gyro axes (CSV) onto the vision-derived body-frame angular velocity
     (vision lifted to the absolute frame with the mocap headset = dev-only ego; a deployable version would use imu0).  A driver that already applied
     P_oxr yields Q ~= D = diag(1,-1,-1) (angle(D^T Q) ~ 0-2 deg); a RAW sensor stream would be offset from D by the factory Rt rotation
     (|angle| ~105 deg => >= ~75 deg from D).  Reported: angle(D^T Q) per recording/controller.
  B (stream-only, needs gyro): median |accel| over samples with |gyro| < 0.5 rad/s (low-rotation, not truly at rest) for each loader variant.
     A stream with a second factory correction / unscaled driver constant reads ~10.2+; the corrected+scaled stream reads ~9.8-10.0.
Also prints the RT factory rotation angle and the angle between (D^T Rt) for the reader's reference."""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import fv_common as C

rows = []
for name in C.ALL8:
    rd = C.rec_dir(name); hm = C.load_mocap_device(rd, "headset")
    for ctrl in C.CTRLS:
        t, g_csv, a_csv = C.raw_csv(rd, ctrl)
        v = C.Vision(C.vision_csv(name), ctrl); ok = v.strong()
        idx = np.flatnonzero(ok)
        Rw = {}
        for i in idx:
            Th = C.MH.world_pose(hm, int(v.ts[i]))
            if Th is not None:
                Rw[i] = Th.R @ v.R[i]
        H = np.zeros((3, 3)); n = 0
        keys = sorted(Rw)
        for a, b in zip(keys[:-1], keys[1:]):
            if b != a + 1 and not ok[a:b + 1].all():
                pass
            dt = (v.ts[b] - v.ts[a]) / 1e9
            if not (0 < dt <= 0.05):
                continue
            wv = Rotation.from_matrix(Rw[a].T @ Rw[b]).as_rotvec() / dt
            if np.linalg.norm(wv) < 1.0 or np.linalg.norm(wv) > 15.0:
                continue
            sel = (t >= v.ts[a]) & (t <= v.ts[b])
            if sel.sum() < 2:
                continue
            wg = g_csv[sel].mean(axis=0)
            H += np.outer(wv, wg); n += 1
        U, S, Vt = np.linalg.svd(H); Q = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        ang = float(np.degrees(np.linalg.norm(Rotation.from_matrix(C.D.T @ Q).as_rotvec())))
        ang_I = float(np.degrees(np.linalg.norm(Rotation.from_matrix(Q).as_rotvec())))
        streams = C.build_streams(ctrl, g_csv, a_csv)
        low = np.linalg.norm(g_csv, axis=1) < 0.5
        row = dict(rec=name, ctrl=ctrl, n_steps=n, angle_Q_to_D_deg=ang, angle_Q_to_I_deg=ang_I, n_low_omega=int(low.sum()))
        for k, (gg, aa) in streams.items():
            row[f"med_absa_lowomega_{k}"] = float(np.median(np.linalg.norm(aa[low], axis=1))) if low.sum() > 50 else np.nan
        row["med_absa_lowomega_csv_raw"] = float(np.median(np.linalg.norm(a_csv[low], axis=1))) if low.sum() > 50 else np.nan
        rows.append(row)
        print(f"{name:13s} {ctrl:16s} steps {n:5d}  angle(Q, D) = {ang:5.2f} deg   angle(Q, I) = {ang_I:6.1f} deg   low-omega samples {int(low.sum()):5d}", flush=True)
df = pd.DataFrame(rows); df.to_csv(C.HERE / "results" / "provenance_stats.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print("\nStatistic B: median |accel| over samples with |gyro|<0.5 rad/s, by loader variant (raw CSV = as recorded):")
print(df[["rec", "ctrl", "n_low_omega", "med_absa_lowomega_csv_raw", "med_absa_lowomega_L0_OLD", "med_absa_lowomega_L1_drop2nd", "med_absa_lowomega_L2_scale_only", "med_absa_lowomega_L3_NEW"]].to_string(index=False))
for ctrl in C.CTRLS:
    c = C.factory_calib(ctrl, 1)
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(c.accel.T_rt.R).as_rotvec()))
    ang_D = np.degrees(np.linalg.norm(Rotation.from_matrix(C.D.T @ c.accel.T_rt.R).as_rotvec()))
    ang_Dt = np.degrees(np.linalg.norm(Rotation.from_matrix(C.D.T @ c.accel.T_rt.R.T).as_rotvec()))
    print(f"{ctrl}: factory accel Rt rotation angle {ang:.1f} deg ; angle(D^T Rt) {ang_D:.1f} deg ; angle(D^T Rt^T) {ang_Dt:.1f} deg")
