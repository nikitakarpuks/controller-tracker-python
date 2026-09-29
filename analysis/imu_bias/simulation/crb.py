"""crb.py -- analytic observability: Cramer-Rao lower bounds on b_g and b_a for a sliding window, using the
REAL trajectories (rotation excitation!) and the REAL vision cadence, white vision noise sigma_rot/sigma_pos.

Gyro   : Z_j = -J_j b - G_j^T e0 + e_j,  J_j = R_j^T int R(s) ds,  G_j = R_0^T R_j        (e0 = nuisance attitude)
Accel  : p_j = p0 + v0 tau_j + P_j - Q_j b_a (+ 1/2 tau_j^2 dg),   Q_j = double integral of R dt   (dg = gravity error)
Vision orientation error in the accel model is NOT included (optimistic bound); the noisy simulation includes it.
"""
import sys, csv, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from simlib import *
from simlib import _IMU

OUT = Path(__file__).parent


def cum(t, x):
    o = np.zeros_like(x); o[1:] = np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(t).reshape((-1,) + (1,) * (x.ndim - 1)), axis=0); return o


def window_crb(tr, tv, W, starts, sig_rot, sig_pos, grav_prior=None):
    """tv: node times (s). Returns list of dict per window start."""
    out = []
    for t0 in starts:
        m = (tv >= t0) & (tv <= t0 + W)
        if m.sum() < 6:
            continue
        tj = tv[m]
        tf = np.arange(tj[0], tj[-1], 0.005)
        R = tr.R(tf)
        C = cum(tf, R)                                  # int R ds (3x3)
        Qm = cum(tf, C)                                 # double integral
        idx = np.clip(np.searchsorted(tf, tj), 0, len(tf) - 1)
        Rj = R[idx]
        Aj = []
        Ag = np.zeros((3 * (len(tj) - 1), 6)); Aa = np.zeros((3 * len(tj), 9 + (3 if grav_prior else 0)))
        for j in range(1, len(tj)):
            Jj = Rj[j].T @ C[idx[j]]
            Gj = Rj[0].T @ Rj[j]
            Ag[3 * (j - 1):3 * j, :3] = -Jj; Ag[3 * (j - 1):3 * j, 3:] = -Gj.T
        for j in range(len(tj)):
            tau = tj[j] - tj[0]
            Aa[3 * j:3 * j + 3, 0:3] = np.eye(3); Aa[3 * j:3 * j + 3, 3:6] = tau * np.eye(3); Aa[3 * j:3 * j + 3, 6:9] = -Qm[idx[j]]
            if grav_prior:
                Aa[3 * j:3 * j + 3, 9:12] = 0.5 * tau ** 2 * np.eye(3)
        Ig = Ag.T @ Ag / sig_rot ** 2 + np.diag([0, 0, 0, 1, 1, 1]) / sig_rot ** 2      # e0 prior
        Ia = Aa.T @ Aa / sig_pos ** 2
        if grav_prior:
            Ia[9:12, 9:12] += np.eye(3) / grav_prior ** 2
        cg = np.linalg.inv(Ig)[:3, :3]; ca = np.linalg.inv(Ia)[6:9, 6:9]
        w = np.linalg.norm(tr.w(tj), axis=1)
        out.append(dict(t0=t0, n=len(tj), sd_bg=np.sqrt(np.trace(cg) / 3), sd_ba=np.sqrt(np.trace(ca) / 3),
                        mean_w=w.mean(),
                        rot_span_deg=np.degrees(np.max([np.linalg.norm(Log(Rj[0].T @ r)) for r in Rj])),
                        sv_min=np.linalg.svd(cum(tf, R)[-1] / (tf[-1] - tf[0]))[1].min()))
    return out


def main():
    rows = []; wrows = []
    for rec, dev in (("static_dark", "right"), ("static_dark", "left"), ("walk_medium", "right")):
        root = REC_ROOT / RECORDINGS[rec] / "mav0"
        t_raw, _, _ = load_imu_csv(root / _IMU[dev] / "data.csv"); T0 = int(t_raw[0] + LAG_NS[dev])
        tr = get_truth(rec, dev, T0)
        tvs_all, strong = load_vision_rows(rec, dev)
        tv_strong = (tvs_all[strong] - T0) / 1e9; tv_all = (tvs_all - T0) / 1e9
        lo, hi = tr.t0 + 1, tr.t1 - 1
        for cadence, tv in (("strong-only (real gaps)", tv_strong), ("all rows (22 ms)", tv_all)):
            tv = tv[(tv > lo) & (tv < hi)]
            for W in (2, 5, 10, 20, 30, 60):
                starts = np.arange(lo + 1, hi - W - 1, max(W / 2, 5.0))
                for gp, gname in ((None, "gravity known"), (0.17, "gravity uncertain 1deg (0.17 m/s^2)")):
                    res = window_crb(tr, tv, W, starts, np.radians(0.4), 4e-3, grav_prior=gp)
                    if not res: continue
                    df = pd.DataFrame(res)
                    if cadence.startswith('strong'):
                        for r_ in res: wrows.append(dict(rec=rec, dev=dev, W=W, gravity=gname, **r_))
                    rows.append(dict(rec=rec, dev=dev, cadence=cadence, W=W, gravity=gname, n_windows=len(df),
                                     sd_bg_median=df.sd_bg.median(), sd_bg_p10=df.sd_bg.quantile(.1), sd_bg_p90=df.sd_bg.quantile(.9),
                                     sd_ba_median=df.sd_ba.median(), sd_ba_p10=df.sd_ba.quantile(.1), sd_ba_p90=df.sd_ba.quantile(.9),
                                     corr_sdba_vs_rotspan=np.corrcoef(df.sd_ba, df.rot_span_deg)[0, 1] if len(df) > 3 else np.nan,
                                     mean_nodes=df.n.mean()))
    pd.DataFrame(wrows).to_csv(OUT / "exp1_crb_windows.csv", index=False)
    df = pd.DataFrame(rows); df.to_csv(OUT / "exp1_crb.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(df[df.cadence.str.startswith("strong")][["rec", "dev", "W", "gravity", "mean_nodes", "sd_bg_median", "sd_ba_median", "corr_sdba_vs_rotspan"]].to_string(index=False, float_format=lambda x: f"{x:.2e}"))


if __name__ == "__main__":
    main()
