"""ab_lever.py -- accelerometer dead-reckoning position error of the LED-origin (the tracked point), for every
combination of  loader variant (L0..L3)  x  lever-arm candidate (a..f), on real recordings.

Truth = the raw VISION position of the LED origin at the end frame (independent of the controller mocap, the bridge
and all lever candidates; vision lifted to the absolute frame with the MOCAP HEADSET pose = dev-only ego).
Start state: vision position/orientation at the start frame, v0 = causal LS-line velocity of the preceding ~60 ms of
vision frames; orientation propagated with the candidate stream's own gyro; a_w = R f_c + g, f_c = f - (alpha x r +
omega x (omega x r)) via the project's _lever_arm_correction (HEAD snapshot).  No parameter is fitted anywhere (all
candidates are fixed vectors / fixed constants), so there is no fitting leakage and every time span is used.

Usage: python ab_lever.py [rec ...]   -> fix_validation/results/ab_<rec>.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import fv_common as C

OUT = C.HERE / "results"
OUT.mkdir(exist_ok=True)
GAPS = (0.1, 0.3, 1.0)


def propagate(R0, g, dts):
    dR = Rotation.from_rotvec(0.5 * (g[:-1] + g[1:]) * dts[:, None]).as_matrix()
    R = np.empty((len(g), 3, 3)); R[0] = R0
    for q in range(len(dts)):
        R[q + 1] = R[q] @ dR[q]
    return R


def integrate(Rs, f_c, v0, p0, dts):
    a_w = np.einsum("nij,nj->ni", Rs, f_c) + C.G_ABS
    v = np.empty_like(a_w); v[0] = v0
    v[1:] = v0 + np.cumsum(0.5 * (a_w[:-1] + a_w[1:]) * dts[:, None], axis=0)
    return p0 + np.sum(0.5 * (v[:-1] + v[1:]) * dts[:, None], axis=0)


def run(name):
    rd = C.rec_dir(name)
    hm = C.load_mocap_device(rd, "headset")
    rows = []
    for ctrl in C.CTRLS:
        t, g_csv, a_csv = C.raw_csv(rd, ctrl)
        streams = C.build_streams(ctrl, g_csv, a_csv)
        levers = C.lever_candidates(ctrl)
        trk = C.vision_track(name, ctrl, hm)
        tsv = trk.ts
        # gap list: fixed gaps (every 3rd frame) + real tracking-loss gaps (consecutive strong frames 0.15-1.5 s apart)
        jobs = []
        for T in GAPS:
            for i in range(0, len(tsv), 3):
                ta = tsv[i]
                kk = np.searchsorted(tsv, ta + int(T * 1e9))
                cs = [k for k in (kk - 1, kk) if 0 <= k < len(tsv) and tsv[k] > ta]
                if not cs:
                    continue
                k = min(cs, key=lambda k: abs(tsv[k] - (ta + T * 1e9)))
                if abs(tsv[k] - (ta + T * 1e9)) > max(0.006, 0.35 * T) * 1e9 or np.max(np.diff(tsv[i:k + 1])) > 0.1e9:
                    continue
                jobs.append(("fixed", T, i, k))
        for i in range(len(tsv) - 1):
            d = (tsv[i + 1] - tsv[i]) / 1e9
            if 0.15 < d <= 1.5:
                jobs.append(("real", d, i, i + 1))
        t0 = time.time()
        for kind, T, i, k in jobs:
            ta, tb = tsv[i], tsv[k]
            i0, i1 = np.searchsorted(t, ta), np.searchsorted(t, tb)
            mid = t[i0:i1][(t[i0:i1] > ta) & (t[i0:i1] < tb)]
            ts_ = np.concatenate(([ta], mid, [tb])).astype(np.int64)
            if len(ts_) < 4:
                continue
            v0 = trk.v_at(ta - int(0.03e9), 0.03, 3)
            if v0 is None:
                continue
            dts = np.diff(ts_) / 1e9
            p0, p_true = trk.P[i], trk.P[k]
            row = dict(rec=name, ctrl=ctrl, kind=kind, T=T, t_start=int(ta), naive=float(np.linalg.norm(p0 + v0 * (tb - ta) / 1e9 - p_true)))
            interp_cache = {}
            for lname, (g_body, a_body) in streams.items():
                gi = np.stack([np.interp(ts_, t, g_body[:, q]) for q in range(3)], 1)
                fi = np.stack([np.interp(ts_, t, a_body[:, q]) for q in range(3)], 1)
                Rs = propagate(trk.R[i], gi, dts)
                for cname, r in levers.items():
                    f_c = fi - C.IH._lever_arm_correction(ts_, ts_, gi, r)
                    row[f"{lname}|{cname}"] = float(np.linalg.norm(integrate(Rs, f_c, v0, p0, dts) - p_true))
                if lname == "L3_NEW":
                    row["omega_rms"] = float(np.sqrt(np.mean(np.sum(gi ** 2, axis=1))))
            rows.append(row)
        print(f"[{name}/{ctrl}] {len(jobs)} gap jobs -> {sum(1 for r in rows if r['ctrl'] == ctrl)} rows ({time.time() - t0:.0f} s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"ab_{name}.csv", index=False)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or C.MODERATE):
        run(nm)
