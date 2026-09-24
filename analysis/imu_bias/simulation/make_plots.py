import sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path(__file__).parent; (OUT / "plots").mkdir(exist_ok=True)

def fig_crb():
    df = pd.read_csv(OUT / "exp1_crb.csv"); df = df[df.cadence.str.startswith("strong")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for (rec, dev), g in df.groupby(["rec", "dev"]):
        gk = g[g.gravity == "gravity known"]; gu = g[g.gravity != "gravity known"]
        ax[0].loglog(gk.W, gk.sd_bg_median, "o-", label=f"{rec}/{dev}")
        ax[1].loglog(gk.W, gk.sd_ba_median, "o-", label=f"{rec}/{dev} gravity known")
        ax[1].loglog(gu.W, gu.sd_ba_median, "s--", label=f"{rec}/{dev} gravity +-1deg")
    for a, refs, lab in ((ax[0], (1e-4, 0.011), "gyro bias (rad/s)"), (ax[1], (0.01, 0.2), "accel bias (m/s^2)")):
        for r in refs: a.axhline(r, color="grey", ls=":", lw=1)
        a.set_xlabel("window length W (s)"); a.set_ylabel("CRB std of " + lab); a.grid(alpha=.3, which="both")
    ax[0].set_title("Gyro bias CRB (vision 0.4deg white); dotted: factory 1e-4 / mocap2gt-scale 0.011")
    ax[1].set_title("Accel bias CRB (vision 4mm, attitude known); dotted: 0.01 / 0.2"); ax[1].legend(fontsize=7)
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "plots" / "fig1_crb.png", dpi=130); plt.close(fig)

def fig_traces():
    for sc in ("large", "drift", "factory"):
        f = OUT / "traces" / f"static_dark_right_{sc}.npz"
        if not f.exists(): continue
        z = np.load(f); tv = z["tv"] - z["tv"][0]
        fig, ax = plt.subplots(2, 3, figsize=(15, 7))
        for c in range(3):
            ax[0, c].plot(tv, z["bgT"][:, c], "k", lw=2, label="truth")
            for n, col in (("A_dt_weighted_tau30", "C1"), ("A_naive_ema_tau30", "C3"), ("C_gyro_W30", "C2"), ("B_q1e-05", "C0")):
                k = n + "__g"
                if k in z: ax[0, c].plot(tv, z[k][:, c], color=col, lw=.8, label=n)
            ax[1, c].plot(tv, z["baT"][:, c], "k", lw=2)
            for n, col in (("A_accel_dt_weighted", "C1"), ("C_accel_W30", "C2"), ("B_q1e-05", "C0")):
                k = n + "__a"
                if k in z: ax[1, c].plot(tv, z[k][:, c], color=col, lw=.8, label=n)
            ax[0, c].set_title(f"gyro bias axis {'xyz'[c]}"); ax[1, c].set_title(f"accel bias axis {'xyz'[c]}"); ax[1, c].set_xlabel("t (s)")
        lo = z["bgT"].min(); span = np.abs(z["bgT"]).max()
        ax[0, 0].legend(fontsize=7); ax[1, 0].legend(fontsize=7)
        for a in ax[0]: a.set_ylim(-max(4 * span, 2e-3), max(4 * span, 2e-3))
        for a in ax[1]: a.set_ylim(-max(4 * np.abs(z["baT"]).max(), 0.05), max(4 * np.abs(z["baT"]).max(), 0.05))
        fig.suptitle(f"static_dark / right / scenario {sc}: truth vs estimates"); fig.tight_layout()
        fig.savefig(OUT / "plots" / f"fig2_traces_{sc}.png", dpi=110); plt.close(fig)

def fig_sens():
    f = OUT / "exp4_sensitivity.csv"
    if not f.exists(): return
    df = pd.read_csv(f); g = df.groupby(["case", "estimator", "kind"]).mean_artifact.mean().reset_index()
    order = list(dict.fromkeys(df.case))
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    for a, kind, unit, refs in ((ax[0], "gyro", "rad/s", (1e-4, 0.011)), (ax[1], "accel", "m/s^2", (0.01, 0.2))):
        for j, est in enumerate(sorted(g[g.kind == kind].estimator.unique())):
            s = g[(g.kind == kind) & (g.estimator == est)].set_index("case").reindex(order)
            a.barh(np.arange(len(order)) + j * 0.27, s.mean_artifact.values, height=0.27, label=est)
        a.set_yticks(np.arange(len(order)) + 0.27); a.set_yticklabels(order, fontsize=8); a.set_xscale("log"); a.invert_yaxis()
        for r in refs: a.axvline(r, color="grey", ls=":")
        a.set_xlabel(f"spurious mean bias (true bias = 0), {unit}"); a.legend(fontsize=8); a.set_title(kind)
    fig.tight_layout(); fig.savefig(OUT / "plots" / "fig3_sensitivity.png", dpi=120); plt.close(fig)

def fig_benefit():
    f = OUT / "exp3_benefit.csv"
    if not f.exists(): return
    df = pd.read_csv(f); df = df[(df.rec == "static_dark") & (df.dev == "right")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for j, sc in enumerate(("factory", "large", "drift")):
        for i, (col, lab) in enumerate((("rot_err_deg_median", "rotation error (deg, median)"), ("pos_err_mm_median", "position error (mm, median)"))):
            for est, ls in (("B_q1e-05", "-"), ("A_dtw30+accel0", "--")):
                for opt, c in (("zero", "k"), ("est", "C0"), ("true", "C2")):
                    s = df[(df.scenario == sc) & (df.estimator == est) & (df.opt == opt)].groupby("horizon_s")[col].mean()
                    if len(s): ax[i, j].loglog(s.index, s.values, ls, color=c, label=f"{est.split('_')[0]} {opt}" if ls == "-" or opt == "est" else None)
            ax[i, j].set_title(f"{sc}: {lab}", fontsize=9); ax[i, j].set_xlabel("horizon (s)"); ax[i, j].grid(alpha=.3, which="both")
    ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "plots" / "fig4_benefit.png", dpi=120); plt.close(fig)

def fig_design():
    f = OUT / "exp5_design.csv"
    if not f.exists(): return
    df = pd.read_csv(f); b = df[(df.estimator == "B") & (df.kind == "gyro")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, sc in zip(ax, ("factory", "large", "drift")):
        g = b[b.scenario == sc].groupby(["q_bg", "meas_infl"]).rms.mean().unstack()
        im = a.imshow(np.log10(g.values), aspect="auto"); a.set_xticks(range(len(g.columns))); a.set_xticklabels(g.columns); a.set_yticks(range(len(g.index))); a.set_yticklabels([f"{x:g}" for x in g.index])
        a.set_xlabel("measurement noise inflation"); a.set_ylabel("q_bg (rad/s/sqrt(s))"); a.set_title(f"B gyro-bias RMS (log10), {sc}")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]): a.text(j, i, f"{g.values[i,j]:.1e}", ha="center", va="center", fontsize=7, color="w")
    fig.tight_layout(); fig.savefig(OUT / "plots" / "fig5_design_B.png", dpi=120); plt.close(fig)

if __name__ == "__main__":
    fig_crb(); fig_traces(); fig_sens(); fig_benefit(); fig_design(); print("plots ok")
