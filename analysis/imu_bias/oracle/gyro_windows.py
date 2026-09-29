import sys, csv
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from gyro_oracle import *
def load(name, ctrl, tau=0.5):
    z = np.load(OUT + f"gyro_terms_{name}_{ctrl}_tau{tau}.npz"); return {k: z[k] for k in z.files}

def window_fit(d, idx, K):
    """bias with K fixed; returns b, SE (from MAD-scaled residual)."""
    r0 = d["r0"][idx]; Jb = d["Jb"][idx]; JK = d["JK"][idx]
    y = -(r0 + JK @ K.reshape(9)); A = Jb.reshape(-1, 3); yy = y.reshape(-1)
    keep = np.ones(len(yy), bool)
    for _ in range(3):
        b, *_ = np.linalg.lstsq(A[keep], yy[keep], rcond=None)
        res = yy - A @ b; s = 1.4826 * np.median(np.abs(res[keep])) + 1e-12
        keep = np.abs(res) < 4 * s
    cov = s**2 * np.linalg.inv(A[keep].T @ A[keep])
    return b, np.sqrt(np.diag(cov))

rows = []; summ = []
for ctrl in CTRLS:
    K = np.load(OUT + f"K_pooled_{ctrl}.npy")
    for name in REC_NAMES:
        d = load(name, ctrl); t = (d["t0"] - d["t0"][0]) / 1e9; n = len(t)
        # per-recording reference (K fixed)
        b_rec, se_rec = window_fit(d, np.arange(n), K)
        for W in (10.0, 20.0, 40.0):
            starts = np.arange(0, t[-1] - W + 1e-9, W / 2); B = []; SE = []; TC = []
            for s0 in starts:
                idx = np.flatnonzero((t >= s0) & (t < s0 + W))
                if len(idx) < 8: continue
                b, se = window_fit(d, idx, K); B.append(b); SE.append(se); TC.append(s0 + W / 2)
                rows.append(dict(rec=name, ctrl=ctrl, W=W, t_center_s=s0 + W / 2, n=len(idx), **{f"b_{a}": b[i] for i, a in enumerate("xyz")}, **{f"se_{a}": se[i] for i, a in enumerate("xyz")}))
            B = np.array(B); SE = np.array(SE); TC = np.array(TC)
            # use non-overlapping windows (every 2nd) for chi2 to avoid overlap correlation
            Bn, SEn = B[::2], SE[::2]
            chi2 = (((Bn - Bn.mean(0)) / SEn) ** 2).sum(0) / max(len(Bn) - 1, 1)
            slope = np.array([np.polyfit(TC, B[:, i], 1)[0] for i in range(3)]) * 60.0   # rad/s per minute
            summ.append(dict(rec=name, ctrl=ctrl, W=W, nwin=len(Bn), sd_x=Bn[:,0].std(), sd_y=Bn[:,1].std(), sd_z=Bn[:,2].std(), se_x=SEn[:,0].mean(), se_y=SEn[:,1].mean(), se_z=SEn[:,2].mean(),
                             chi2_x=chi2[0], chi2_y=chi2[1], chi2_z=chi2[2], slope_x=slope[0], slope_y=slope[1], slope_z=slope[2]))
with open(OUT + "gyro_bias_windows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(OUT + "gyro_bias_window_stats.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
for W in (10.0, 20.0, 40.0):
    for ctrl in CTRLS:
        sub = [s for s in summ if s["W"] == W and s["ctrl"] == ctrl]
        print(f"W={W:4.0f}s {ctrl[:5]}: mean SE/axis {np.mean([[s['se_x'],s['se_y'],s['se_z']] for s in sub],0).round(4)}  window scatter sd {np.mean([[s['sd_x'],s['sd_y'],s['sd_z']] for s in sub],0).round(4)}  chi2/dof (=1 => const bias) {np.mean([[s['chi2_x'],s['chi2_y'],s['chi2_z']] for s in sub],0).round(2)}  slope mean {np.mean([[s['slope_x'],s['slope_y'],s['slope_z']] for s in sub],0).round(4)} rad/s/min")
