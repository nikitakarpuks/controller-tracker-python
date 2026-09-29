"""ab_factory_variants.py -- is any OTHER way of reading the factory accel Rt/translation competitive with -R^T t?  L3_NEW stream, 3 recordings."""
import numpy as np, pandas as pd
import fv_common as C
from ab_lever import propagate, integrate
recs = ["static_dark", "walk_easy", "walk_medium"]
rows = []
for name in recs:
    rd = C.rec_dir(name); hm = C.load_mocap_device(rd, "headset")
    for ctrl in C.CTRLS:
        t, g_csv, a_csv = C.raw_csv(rd, ctrl)
        g_body, a_body = C.build_streams(ctrl, g_csv, a_csv)["L3_NEW"]
        c = C.factory_calib(ctrl, 1); a = c.accel.T_rt
        cand = {"zero": np.zeros(3), "t (today)": a.t, "-R^T t  [proposed]": -a.R.T @ a.t, "R^T t": a.R.T @ a.t, "-R t": -a.R @ a.t, "R t": a.R @ a.t,
                "D(-R^T t)": C.D @ (-a.R.T @ a.t), "D t": C.D @ a.t}
        trk = C.vision_track(name, ctrl, hm); tsv = trk.ts
        for T in (0.3, 1.0):
            for i in range(0, len(tsv), 3):
                ta = tsv[i]; kk = np.searchsorted(tsv, ta + int(T * 1e9)); cs = [k for k in (kk - 1, kk) if 0 <= k < len(tsv) and tsv[k] > ta]
                if not cs: continue
                k = min(cs, key=lambda k: abs(tsv[k] - (ta + T * 1e9)))
                if abs(tsv[k] - (ta + T * 1e9)) > max(0.006, 0.35 * T) * 1e9 or np.max(np.diff(tsv[i:k + 1])) > 0.1e9: continue
                tb = tsv[k]; i0, i1 = np.searchsorted(t, ta), np.searchsorted(t, tb)
                mid = t[i0:i1][(t[i0:i1] > ta) & (t[i0:i1] < tb)]; ts_ = np.concatenate(([ta], mid, [tb])).astype(np.int64)
                v0 = trk.v_at(ta - int(0.03e9), 0.03, 3)
                if len(ts_) < 4 or v0 is None: continue
                dts = np.diff(ts_) / 1e9
                gi = np.stack([np.interp(ts_, t, g_body[:, q]) for q in range(3)], 1); fi = np.stack([np.interp(ts_, t, a_body[:, q]) for q in range(3)], 1)
                Rs = propagate(trk.R[i], gi, dts); row = dict(rec=name, ctrl=ctrl, T=T, t_start=int(ta))
                for nm, r in cand.items():
                    row[nm] = float(np.linalg.norm(integrate(Rs, fi - C.IH._lever_arm_correction(ts_, ts_, gi, r), v0, trk.P[i], dts) - trk.P[k]) * 1000)
                rows.append(row)
df = pd.DataFrame(rows); df.to_csv(C.HERE / "results" / "ab_factory_variants.csv", index=False)
cols = [c for c in df.columns if c not in ("rec", "ctrl", "T", "t_start")]
for ctrl in C.CTRLS:
    print(ctrl)
    for T in (0.3, 1.0):
        s = df[(df.ctrl == ctrl) & (df["T"] == T)]
        print(f"  T={T}s (n={len(s)}): " + "  ".join(f"{c}: {s[c].median():6.1f}" for c in cols))
