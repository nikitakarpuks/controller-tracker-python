"""Step 4a: DIRECT payoff test. Fit (b) / (K) / (b,K) on the FIRST 60 s ONLY (vision steps, body-frame per-step LS), then
measure gyro-integrated rotation-prediction error over gaps on the REST of the recording. Correction model:
    omega_corrected = (I+K)^-1 (omega_meas - b)        [first-order model from e0 = -dt (b + K omega)]
Reference rotation = vision relative rotation with mocap headset ego-motion removed. Errors in degrees."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *
from step3f_scale_misalign import fit_body

GAPS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

def gap_err_gyro(run, ctrl, prep, gyro, gap_s, t_lo, t_hi, stride=3):
    d = run.ctrl[ctrl]; ts, R, ok = prep["ts"], prep["R_wc"], prep["ok"]
    idx = np.flatnonzero(ok); tt = ts[idx]
    tol = max(0.006, 0.35 * gap_s) * 1e9
    out = []
    for k in range(0, len(idx), stride):
        a = idx[k]
        if ts[a] < t_lo or ts[a] > t_hi: continue
        kk = np.searchsorted(tt, ts[a] + gap_s * 1e9)
        cand = [c for c in (kk - 1, kk) if 0 <= c < len(tt) and tt[c] > ts[a]]
        if not cand: continue
        c = min(cand, key=lambda c: abs(tt[c] - (ts[a] + gap_s * 1e9)))
        if abs(tt[c] - (ts[a] + gap_s * 1e9)) > tol: continue
        b = idx[c]
        Rg = integrate_gyro_segment(d["t"], gyro, int(ts[a]), int(ts[b]))
        if Rg is None: continue
        out.append(rot_deg(Rg.T @ (R[a].T @ R[b])))
    return np.array(out)

def corrected(gyro, b, K):
    return ((np.linalg.inv(np.eye(3) + K) @ (gyro - b).T)).T

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
    run = Run(name)
    for c in CTRLS:
        d = run.ctrl[c]; prep = prepare(run, c, ego="mocap"); steps, _ = make_steps(run, c, prep)
        t0 = prep["ts"][prep["ok"]][0]; split = t0 + int(60e9); t_end = prep["ts"][-1]
        train = [s for s in steps if s["t"] < split]
        beta_bK, se, _ = fit_body(train)                     # b + K (12)
        b_bK, K_bK = beta_bK[:3], beta_bK[3:].reshape(3, 3)
        # b only (body LS): b = -sum dt e0 / sum dt^2
        b_only = -sum(s["dt"] * s["e0"] for s in train) / sum(s["dt"] ** 2 for s in train)
        # K only
        beta_K, _, _ = fit_body(train, use_b=False); K_only = beta_K.reshape(3, 3)
        # isotropic scale only
        num = -sum(s["dt"] * (s["e0"] @ s["omega"]) for s in train); den = sum(s["dt"] ** 2 * (s["omega"] @ s["omega"]) for s in train)
        k_iso = num / den
        variants = {
            "zero": d["gyro"],
            "b only (fit<60s)": corrected(d["gyro"], b_only, np.zeros((3, 3))),
            "isotropic scale only": corrected(d["gyro"], np.zeros(3), k_iso * np.eye(3)),
            "K only (9)": corrected(d["gyro"], np.zeros(3), K_only),
            "b + K (12)": corrected(d["gyro"], b_bK, K_bK),
        }
        print(f"\n[{name}/{c}] fit on first 60 s ({len(train)} steps):  b_only={np.round(b_only,4)}  k_iso={k_iso:+.4f}  b(bK)={np.round(b_bK,4)}")
        print(f"      K(bK)=\n{np.round(K_bK,4)}")
        for region, (lo, hi) in (("TEST (t>60s)", (split, t_end)), ("train (t<60s, in-sample)", (t0, split))):
            print(f"   --- {region}: median / p90 rotation-prediction error (deg) by gap")
            print("   gap(s):                " + "".join(f"{g:>16.2f}" for g in GAPS))
            for lab, gy in variants.items():
                cells = []
                for g in GAPS:
                    e = gap_err_gyro(run, c, prep, gy, g, lo, hi)
                    cells.append(f"{np.median(e):.2f}/{np.percentile(e,90):.2f}(n{len(e)})" if len(e) > 10 else "   n/a")
                print(f"   {lab:22s}" + "".join(f"{x:>16s}" for x in cells))
