"""
Fit pixel_threshold model:  threshold = margin * (a * facing_cos/depth_m² + c * velocity_px + b)

Reads calibration data written by main.py's `debug.calibration_csv` (per-LED
depth_m / facing_cos / velocity_px / brightness rows, with depth_m and
facing_cos computed from the pose actually solved that frame — see
TrackingSystem.solved_led_geometry — not the pre-match extrapolated
prediction, which drifts during fast rotation/acceleration).

`brightness` is a matched blob's peak pixel value, not a threshold that was
ever applied — so we fit (a, c, b) to predict that peak, then apply a safety
margin below it to get the actual pixel_threshold coefficients to paste into
config.yml. Using the raw brightness fit directly as the threshold would set
it equal to the LED's own peak, which would fail to detect that same LED on
the next, slightly dimmer frame.

Usage: python data/fit_pixel_threshold.py [path/to/calibration.csv]
(defaults to ./data/calib_warm.csv, matching config.yml's example path)
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir("/home/nikitakarpuks/PyCharmProjects/controller-tracker-python")

CSV_PATH      = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data/calib_warm.csv")
SAFETY_MARGIN = 0.6   # pixel_threshold = margin * predicted peak brightness

# ── Load ────────────────────────────────────────────────────────────────────
if not CSV_PATH.exists():
    raise SystemExit(
        f"Calibration CSV not found: {CSV_PATH}\n"
        f"Set debug.calibration_csv in config.yml and run a tracking session first."
    )

depths, cos_vals, vel_px, brightness, labels = [], [], [], [], []
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        depths.append(float(row["depth_m"]))
        cos_vals.append(float(row["facing_cos"]))
        vel_px.append(float(row.get("velocity_px") or 0.0))
        brightness.append(float(row["brightness"]))
        labels.append(f"{row['ctrl_name']}/cam{row['cam_idx']}/led{row['led_id']}")

depths     = np.array(depths)
cos_vals   = np.array(cos_vals)
vel_px     = np.array(vel_px)
brightness = np.array(brightness, dtype=float)

if len(depths) < 4:
    raise SystemExit(f"Only {len(depths)} calibration rows in {CSV_PATH} — "
                      f"need more (and more varied) samples to fit 3 params.")

# ── Fit: peak brightness ≈ a * cos/d² + c * v_px + b ─────────────────────────
x_geom = cos_vals / depths ** 2
A = np.column_stack([x_geom, vel_px, np.ones_like(x_geom)])
(a_raw, c_raw, b_raw), *_ = np.linalg.lstsq(A, brightness, rcond=None)

cond = np.linalg.cond(A)
if cond > 1e4:
    print(f"WARNING: design matrix condition number {cond:.1e} is high — "
          f"cos/d² and velocity_px may not vary independently enough in this "
          f"dataset to separate their effects reliably (collect more spread).")

a_thr, c_thr, b_thr = SAFETY_MARGIN * a_raw, SAFETY_MARGIN * c_raw, SAFETY_MARGIN * b_raw

print("=" * 68)
print(f"  peak brightness  ≈ {a_raw:.5f} * cos/d²  +  {c_raw:.5f} * v_px  +  {b_raw:.5f}")
print(f"  pixel_threshold  = {SAFETY_MARGIN} * peak brightness")
print(f"                   = {a_thr:.5f} * cos/d²  +  {c_thr:.5f} * v_px  +  {b_thr:.5f}")
print("=" * 68)
print("\nPaste into config.yml pose_guided_thresholds.pixel_threshold:")
print(f"    a: {a_thr:.5f}")
print(f"    c: {c_thr:.5f}")
print(f"    b: {b_thr:.5f}")

pred = a_raw * x_geom + c_raw * vel_px + b_raw
n_show = min(20, len(labels))
print(f"\n{'Sample':<28} {'measured':>9} {'predicted':>10} {'error':>7}")
print("-" * 60)
for label, br, p in list(zip(labels, brightness, pred))[:n_show]:
    print(f"  {label:<26} {br:>9.1f} {p:>10.1f} {p - br:>+7.1f}")
if len(labels) > n_show:
    print(f"  ... ({len(labels) - n_show} more rows)")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle(
    f"peak brightness = {a_raw:.4f}·cos/d² + {c_raw:.4f}·v_px + {b_raw:.4f}   "
    f"(pixel_threshold = {SAFETY_MARGIN}× that)", fontsize=11,
)

ax = axes[0]
x_line = np.linspace(0, max(x_geom.max() * 1.1, 1e-6), 300)
ax.scatter(x_geom, brightness, s=10, alpha=0.5)
ax.plot(x_line, a_raw * x_line + c_raw * vel_px.mean() + b_raw,
        color="steelblue", label="fit @ mean v_px")
ax.set_xlabel("cos / depth_m²")
ax.set_ylabel("brightness")
ax.set_title("Regression space (geometry)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
d_range = np.linspace(max(depths.min(), 0.05), max(depths.max(), 0.06), 300)
for cos_fixed, color in [(1.0, "steelblue"), (0.7, "orange"), (0.4, "tomato")]:
    ax.plot(d_range, a_raw * cos_fixed / d_range ** 2 + c_raw * vel_px.mean() + b_raw,
            color=color, label=f"cos={cos_fixed}")
ax.scatter(depths, brightness, s=10, alpha=0.4, color="grey")
ax.axhline(b_raw, color="grey", linestyle="--", linewidth=0.8, label=f"floor b={b_raw:.1f}")
ax.set_xlabel("depth_m")
ax.set_ylabel("brightness")
ax.set_title("Brightness vs depth")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2]
cos_range = np.linspace(0.1, 1.0, 300)
for depth_fixed, color in [(0.1, "tomato"), (0.3, "orange"), (0.6, "steelblue"), (1.0, "green")]:
    ax.plot(cos_range, a_raw * cos_range / depth_fixed ** 2 + c_raw * vel_px.mean() + b_raw,
            color=color, label=f"d={depth_fixed}m")
ax.scatter(cos_vals, brightness, s=10, alpha=0.4, color="grey")
ax.set_xlabel("facing_cos")
ax.set_ylabel("brightness")
ax.set_title("Brightness vs angle")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[3]
resid_geom = brightness - (a_raw * x_geom + b_raw)  # geometry term removed
ax.scatter(vel_px, resid_geom, s=10, alpha=0.5)
v_line = np.linspace(0, max(vel_px.max(), 1e-6), 300)
ax.plot(v_line, c_raw * v_line, color="steelblue", label=f"c={c_raw:.4f}")
ax.set_xlabel("velocity_px")
ax.set_ylabel("brightness residual (geometry removed)")
ax.set_title("Velocity effect")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./data/fit_pixel_threshold.png", dpi=150)
plt.show()
print("\nPlot saved → data/fit_pixel_threshold.png")
