"""Step 3b: what else besides bias explains the gyro-vs-mocap per-step residual?  Weighted LS in the WORLD frame:
   sum_j -R_j e0_j  =  sum_j dt_j R_j ( b + K omega_j )      (b: 3 params, K: 3x3 gyro scale/misalignment)
Nested models on WHOLE recording (mocap, low-noise), then windowed stability of b with and without K."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *

name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)

def design(steps, use_K, use_timing=False):
    rows, y = [], []
    for k, s in enumerate(steps):
        Rj, dt, om = s["R_end"], s["dt"], s["omega"]
        cols = [dt * Rj]                                            # b
        if use_K:
            # K omega with K row-major: (K omega)_a = sum_b K_ab om_b  -> R_j * [om_b e_a]
            blk = np.zeros((3, 9))
            for a in range(3):
                for b in range(3):
                    blk[:, a * 3 + b] = Rj[:, a] * om[b] * dt
            cols.append(blk)
        if use_timing:                                              # gyro time-shift term: e ~ dt*(omega_{k+1}-omega_k)*delta -> skip here
            pass
        rows.append(np.hstack(cols)); y.append(-(Rj @ s["e0"]))
    return np.vstack(rows), np.concatenate(y)

for c in CTRLS:
    st = mocap_steps(run, c)
    print(f"\n[{name}/{c}] mocap steps {len(st)}   median |omega| {np.median([s['rate'] for s in st]):.2f} rad/s")
    tot = np.concatenate([-(s['R_end'] @ s['e0']) for s in st])
    print(f"   world-frame accumulated residual  |sum e|  : {np.linalg.norm(np.sum([-(s['R_end'] @ s['e0']) for s in st],0)):.4f} rad over {sum(s['dt'] for s in st):.1f}s  "
          f"(if pure bias 0.01 rad/s -> {0.01*sum(s['dt'] for s in st):.2f} rad)")
    for use_K, label in ((False, "bias only (3)"), (True, "bias + K (12)")):
        X, y = design(st, use_K)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        print(f"   {label:15s}: residual norm {np.linalg.norm(res):.4f} (from {np.linalg.norm(y):.4f}); b = {np.round(beta[:3],5)} rad/s")
        if use_K:
            K = beta[3:].reshape(3, 3)
            print(f"      K (scale/misalignment; rows=out axis):\n{np.round(K,4)}\n      diag scale err {np.round(np.diag(K)*100,3)} %   max off-diag {np.abs(K-np.diag(np.diag(K))).max()*100:.3f} % ({np.degrees(np.abs(K-np.diag(np.diag(K))).max()):.2f} deg)")
    # windowed stability with/without K removal
    Xf, yf = design(st, True); beta_f, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
    Kf = beta_f[3:].reshape(3, 3)
    tt = np.array([s["t"] for s in st])
    for W in (10, 20):
        bs_plain, bs_K = [], []
        cen = np.arange(tt[0] + int(W / 2 * 1e9), tt[-1] - int(W / 2 * 1e9), int(2e9))
        for cc in cen:
            lo, hi = np.searchsorted(tt, cc - int(W / 2 * 1e9)), np.searchsorted(tt, cc + int(W / 2 * 1e9))
            sub = st[lo:hi]
            A = sum(s["dt"] * s["R_end"] for s in sub) + 1e-2 * np.eye(3)
            S = sum(-(s["R_end"] @ s["e0"]) for s in sub)
            S2 = S - sum(s["dt"] * (s["R_end"] @ (Kf @ s["omega"])) for s in sub)
            bs_plain.append(np.linalg.solve(A, S)); bs_K.append(np.linalg.solve(A, S2))
        bs_plain, bs_K = np.array(bs_plain), np.array(bs_K)
        print(f"   window {W}s: std over time of windowed b  plain {np.round(bs_plain.std(0),5)}   after removing global K*omega {np.round(bs_K.std(0),5)}   "
              f"(mean plain {np.round(bs_plain.mean(0),4)} / K-removed {np.round(bs_K.mean(0),4)})")
