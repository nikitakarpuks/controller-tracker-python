"""make_figures.py -- figures for REPORT.md, built ONLY from the saved result CSVs (no recomputation)."""
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
R = D / "results_csv" if (D / "results_csv").exists() else D
T = D / "trajectories" if (D / "trajectories").exists() else D
F = D / "figures"; F.mkdir(exist_ok=True)
CTRLS = ("left_controller", "right_controller")


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def series(rows, **match):
    out = {}
    for r in rows:
        if all(r[k] == str(v) for k, v in match.items()):
            g = r.get("gap", r.get("gap_s"))
            if g == "real":            # 'real tracking-loss gaps' class is reported in the tables, not on the gap axis
                continue
            out[float(g)] = float(r["median_deg"] if "median_deg" in r else r["median_m"])
    return dict(sorted(out.items()))


# ---------------- Fig 1: gyro rotation-prediction error vs gap ----------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
cases = [("static_dark", "gyro_estimators_fast_static_dark_mocap_own60.csv", "K fit on own first 60 s"),
         ("walk_medium", "gyro_estimators_fast_walk_medium_mocap_static_dark.csv", "K fit on static_dark (cross-recording)")]
for col, (rec, fn, ktxt) in enumerate(cases):
    p = R / fn
    if not p.exists():
        continue
    rows = read(p)
    for row, c in enumerate(CTRLS):
        ax = axes[row, col]
        z = series(rows, ctrl=c, gyro_variant="rawgyro", family="zero")
        k = series(rows, ctrl=c, gyro_variant="gyro+K", family="zero")
        ax.plot(list(z), list(z.values()), "o-", color="#c0392b", label="factory gyro, bias 0 (today)")
        ax.plot(list(k), list(k.values()), "s-", color="#2471a3", label="+ static K (scale/misalignment)")
        for fam, col_, lab in (("worldRLS", "#27ae60", "+ K + causal bias (worldRLS)"), ("feedbackEMA", "#8e44ad", "+ K + causal bias (feedbackEMA)")):
            s = series(rows, ctrl=c, gyro_variant="gyro+K", family=fam)
            if s:
                ax.plot(list(s), list(s.values()), "^--", color=col_, label=lab, alpha=0.85)
        ax.set_title(f"{rec} / {c.split('_')[0]}   ({ktxt})", fontsize=9)
        ax.set_xscale("log"); ax.grid(alpha=0.3)
        if col == 0: ax.set_ylabel("median rotation-prediction error (deg)")
        if row == 1: ax.set_xlabel("gap length (s)")
axes[0, 0].legend(fontsize=7.5)
fig.suptitle("Gyro: static K is the big lever; a live bias adds little (held-out, t>60 s)")
fig.tight_layout(); fig.savefig(F / "fig1_gyro_gap_error.png", dpi=140); plt.close(fig)

# ---------------- Fig 2: accel position error budget vs gap ----------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for col, rec in enumerate(("static_dark", "walk_medium")):
    fac = R / f"accel_causal_{rec}.csv"; brg = R / f"accel_causal_{rec}_leverBridge.csv"
    if not (fac.exists() and brg.exists()):
        continue
    rf, rb = read(fac), read(brg)
    for row, c in enumerate(CTRLS):
        ax = axes[row, col]
        def s(rows, m):
            d = {}
            for r in rows:
                if r["ctrl"] == c and r["v0_source"] == "vision" and r["method"] == m:
                    d[float(r["gap_s"])] = float(r["median_m"])
            return dict(sorted(d.items()))
        for lab, rows, m, colr, mk in (("naive const-velocity", rb, "naive", "#7f8c8d", "x"),
                                       ("accel, factory lever (main.py today)", rf, "accel b=0", "#c0392b", "o"),
                                       ("accel, bridge lever", rb, "accel b=0", "#2471a3", "s"),
                                       ("+ const b,K_a (first 60 s)", rb, "const b+K_a (first 60 s)", "#27ae60", "^")):
            d = s(rows, m)
            if d: ax.plot(list(d), list(d.values()), mk + "-", color=colr, label=lab)
        ax.set_title(f"{rec} / {c.split('_')[0]}", fontsize=9); ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        if col == 0: ax.set_ylabel("median position error (m)")
        if row == 1: ax.set_xlabel("gap length (s)")
axes[0, 0].legend(fontsize=7.5)
fig.suptitle("Accel: fixing the lever arm is the largest gain; static K_a next; bias is small (held-out, t>60 s, v0 from vision)")
fig.tight_layout(); fig.savefig(F / "fig2_accel_gap_error.png", dpi=140); plt.close(fig)

# ---------------- Fig 3: gyro bias trajectories (static_dark, K-corrected) vs mocap oracle ----------------
orc = read(R / "oracle_gyro_bias_static_dark.csv") if (R / "oracle_gyro_bias_static_dark.csv").exists() else []
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
for row, c in enumerate(CTRLS):
    fn = T / f"traj_static_dark_{c}_gyro+K_mocap_own60_worldRLS.csv"
    tr = read(fn) if fn.exists() else []
    o = [r for r in orc if r["ctrl"] == c]
    for ax_i, k in enumerate("xyz"):
        ax = axes[row, ax_i]
        if o: ax.plot([float(r["t_s"]) for r in o], [float(r[f"or_{k}"]) for r in o], color="#7f8c8d", label="mocap window oracle (10 s, raw gyro)")
        if tr: ax.plot([float(r["t_s"]) for r in tr], [float(r[f"b{k}"]) for r in tr], color="#27ae60", label="causal worldRLS (K-corrected gyro)")
        ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3); ax.set_title(f"{c.split('_')[0]} b_g[{k}] (rad/s)", fontsize=9)
        ax.set_ylim(-0.08, 0.08)
axes[0, 0].legend(fontsize=7); fig.suptitle("static_dark gyro bias: causal estimate vs mocap oracle (extra bias on top of the factory correction)")
fig.tight_layout(); fig.savefig(F / "fig3_gyro_bias_trajectories.png", dpi=140); plt.close(fig)

# ---------------- Fig 4: accel bias trajectories (RLS, bridge lever) ----------------
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
for row, c in enumerate(CTRLS):
    fn = T / f"traj_accel_static_dark_{c}_leverBridge.csv"
    if not fn.exists(): continue
    tr = read(fn)
    for ax_i, k in enumerate("xyz"):
        ax = axes[row, ax_i]; ax.plot([float(r["t_s"]) for r in tr], [float(r[f"b{k}"]) for r in tr], color="#2471a3")
        ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3); ax.set_title(f"{c.split('_')[0]} b_a[{k}] (m/s^2), b-only RLS", fontsize=9)
fig.suptitle("static_dark accel bias, causal window-RLS (b only, bridge lever): looks large because it also absorbs K_a x gravity")
fig.tight_layout(); fig.savefig(F / "fig4_accel_bias_trajectories.png", dpi=140); plt.close(fig)
print("figures written to", F)
