import sys, csv, pickle
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
OUT = "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle/"
rows = list(csv.DictReader(open(OUT + "gyro_bias_windows.csv"))); REC = ["static_dark","walk_dark","static_easy","static_medium","static_hard","walk_easy","walk_medium","walk_hard"]
fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
for r_i, ctrl in enumerate(("left_controller", "right_controller")):
    for a_i, a in enumerate("xyz"):
        for rec in REC:
            s = [r for r in rows if r["ctrl"] == ctrl and r["rec"] == rec and float(r["W"]) == 20.0]
            ax[r_i, a_i].plot([float(r["t_center_s"]) for r in s], [float(r["b_" + a]) for r in s], lw=0.9, alpha=0.7, label=rec)
        ax[r_i, a_i].axhline(0, c="k", lw=0.5); ax[r_i, a_i].set_title(f"{ctrl[:5]} gyro bias {a} (mocap oracle, 20 s windows, pooled K)")
        ax[r_i, a_i].axhspan(-1e-4, 1e-4, color="r", alpha=0.6, label="factory BiasUncertainty (1e-4)")
ax[1, 0].set_xlabel("time in recording (s)"); ax[0, 0].set_ylabel("rad/s"); ax[0, 0].legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(OUT + "fig_gyro_bias_windows.png", dpi=110); plt.close(fig)
ub = list(csv.DictReader(open(OUT + "gyro_upperbound.csv")))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for v in ("V0_factory", "V1_const_bias_only", "V3_K_only", "V2_const_bias+K", "V4_timevarying_bias+K(10s)"):
    s = [r for r in ub if r["variant"] == v and r["scope"] == "all"]
    ax[0].plot([float(r["T_s"]) for r in s], [float(r["median_deg"]) for r in s], "o-", label=v)
ax[0].set_xscale("log"); ax[0].set_xlabel("gap T (s)"); ax[0].set_ylabel("median rotation error (deg)"); ax[0].legend(fontsize=7); ax[0].set_title("Gyro prediction vs mocap, oracle corrections (all recs)")
au = list(csv.DictReader(open(OUT + "accel_upperbound.csv")))
for v in sorted({r["variant"] for r in au}):
    s = [r for r in au if r["variant"] == v]; ax[1].plot([float(r["T_s"]) for r in s], [float(r["median_mm"]) for r in s], "o-", label=v)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("gap T (s)"); ax[1].set_ylabel("median position error (mm)"); ax[1].legend(fontsize=7); ax[1].set_title("Accel dead-reckoning vs mocap, oracle corrections (6 moderate recs)")
fig.tight_layout(); fig.savefig(OUT + "fig_upper_bounds.png", dpi=110); plt.close(fig)
ab = list(csv.DictReader(open(OUT + "accel_bias_blocks.csv")))
fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for r_i, ctrl in enumerate(("left_controller", "right_controller")):
    for a_i, a in enumerate("xyz"):
        for rec in ["static_dark","walk_dark","static_easy","static_medium","walk_easy","walk_medium"]:
            s = [r for r in ab if r["ctrl"] == ctrl and r["rec"] == rec]
            ax[r_i, a_i].errorbar([float(r["t_center_s"]) for r in s], [float(r["b_" + a]) for r in s], [float(r["se_" + a]) for r in s], lw=0.9, alpha=0.7, label=rec, capsize=2)
        ax[r_i, a_i].axhline(0, c="k", lw=0.5); ax[r_i, a_i].set_title(f"{ctrl[:5]} accel bias {a} (20 s blocks, pooled S,dg)")
ax[0, 0].legend(fontsize=6); ax[0, 0].set_ylabel("m/s^2"); fig.tight_layout(); fig.savefig(OUT + "fig_accel_bias_blocks.png", dpi=110); plt.close(fig)
print("figs ok")
