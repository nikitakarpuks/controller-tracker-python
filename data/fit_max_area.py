"""
Fit max_area model:  max_area = a * facing_cos / depth_m² + c * velocity_px + b

Upper-bound fit: 'a' and 'c' (the geometry and velocity slopes) come from an
ordinary least-squares fit — the best-fit trend through the middle of the
data — then 'b' is set so UPPER_PERCENTILE % of non-outlier blobs fall BELOW
the resulting curve, i.e. no real LED blob gets filtered. Outliers (oversized
merged blobs) are excluded via IQR before fitting.

velocity_px is motion blur's effect: a moving LED smears into a longer,
larger blob, roughly linearly in pixel displacement (see the "stadium shape"
derivation — area ≈ π·r² + 2·r·d for blur length d) — additive, not
multiplicative, same reasoning as pixel_threshold's own velocity term in
fit_pixel_threshold.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH      = "./calib_warm_pass1.csv"
UPPER_PCTILE  = 99.9   # % of clean data that must lie below the curve

# Set to (a, c, b) to skip fitting entirely and just check hand-picked values
# against the same plots/coverage stat below — e.g. MANUAL_ABC = (0.9702, 0.0, 15.9131)
MANUAL_ABC = (1.2, 0.1544, 20)

# Set False to drop facing_cos from the model entirely: max_area = a/depth² + c*v_px + b.
# Useful when the area-vs-angle plot shows no discernible trend — just noise
# spread across all cos values — rather than a real (if noisy) shape.
USE_FACING_COS = False

# ── Load & prepare ────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")

depths   = df["depth_m"].values
cos_vals = df["facing_cos"].values
vel_vals = df["velocity_px"].values
areas    = df["area"].values
x = (cos_vals / depths ** 2) if USE_FACING_COS else (1.0 / depths ** 2)
formula_str = "cos / d²" if USE_FACING_COS else "1 / d²"

# ── Remove outliers (large merged blobs) via IQR on raw areas ────────────────
q1, q3    = np.percentile(areas, [1, 99])
iqr       = q3 - q1
upper_cut = q3 + 8.0 * iqr
clean     = areas <= upper_cut

x_c, vel_c, areas_c = x[clean], vel_vals[clean], areas[clean]
print(f"After outlier removal: {clean.sum()} / {len(areas)} rows kept "
      f"(cut at area > {upper_cut:.1f} px²)")

if MANUAL_ABC is not None:
    # ── Skip fitting — verify hand-picked values against the data instead ────
    a, c, b = MANUAL_ABC
    bound_label = "manual bound"
else:
    # ── Fit slopes 'a' (geometry) and 'c' (velocity) with least squares ──────
    A = np.column_stack([x_c, vel_c, np.ones_like(x_c)])
    (a, c, _), *_ = np.linalg.lstsq(A, areas_c, rcond=None)

    # ── Set 'b' so UPPER_PCTILE % of clean data lies below the curve ────────
    residuals_c = areas_c - a * x_c - c * vel_c
    b = float(np.percentile(residuals_c, UPPER_PCTILE))
    bound_label = f"{UPPER_PCTILE:.0f}th pctile bound"

below = (areas_c <= a * x_c + c * vel_c + b).mean() * 100
print("=" * 68)
if MANUAL_ABC is not None:
    print(f"  max_area = {a:.5f} * {formula_str}  +  {c:.5f} * v_px  +  {b:.5f}   (manual, unfit)")
    print(f"  {below:.1f}% of clean blobs fall below this curve "
          f"(target was {UPPER_PCTILE:.0f}%)")
else:
    print(f"  max_area = {a:.5f} * {formula_str}  +  {c:.5f} * v_px  +  {b:.5f}   (fitted)")
    print(f"  {below:.1f}% of clean blobs fall below the curve")
print("=" * 68)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle(
    f"max_area = {a:.4f} · {formula_str}  +  {c:.4f} · v_px  +  {b:.4f}   "
    f"({below:.0f}% below, {bound_label}, n={clean.sum()})",
    fontsize=11,
)

# 1. Regression space (geometry, at mean velocity)
ax = axes[0]
x_line = np.linspace(0, x_c.max() * 1.05, 300)
ax.scatter(x[~clean], areas[~clean], s=12, color="lightgrey", alpha=0.6, label="outliers (excluded)")
ax.scatter(x_c, areas_c,            s=8,  alpha=0.4,          label="clean observations")
ax.plot(x_line, a * x_line + c * vel_c.mean() + b, color="tomato", linewidth=1.5,
        label=f"{bound_label} @ mean v_px")
ax.set_xlabel(formula_str.replace("d²", "depth_m²"))
ax.set_ylabel("area (px²)")
ax.set_title("Regression space")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Area vs depth (at mean velocity)
ax = axes[1]
d_range = np.linspace(0.05, 1.0, 300)
ax.scatter(depths[~clean], areas[~clean], s=12, color="lightgrey", alpha=0.6)
ax.scatter(depths[clean],  areas[clean],  s=8,  alpha=0.4)
if USE_FACING_COS:
    for cos_fixed, color, lbl in [(1.0, "steelblue", "cos=1.0"), (0.7, "orange", "cos=0.7"), (0.4, "tomato", "cos=0.4")]:
        ax.plot(d_range, a * cos_fixed / d_range ** 2 + c * vel_c.mean() + b, color=color, label=lbl)
else:
    ax.plot(d_range, a / d_range ** 2 + c * vel_c.mean() + b, color="steelblue", label="model (cos-independent)")
ax.axhline(b, color="grey", linestyle="--", linewidth=0.8, label=f"floor b={b:.1f}")
ax.set_xlabel("depth_m")
ax.set_ylabel("area (px²)")
ax.set_title("Area vs depth")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3. Area vs cos (at mean velocity)
ax = axes[2]
cos_range = np.linspace(0.1, 1.0, 300)
ax.scatter(cos_vals[~clean], areas[~clean], s=12, color="lightgrey", alpha=0.6)
ax.scatter(cos_vals[clean],  areas[clean],  s=8,  alpha=0.4)
for depth_fixed, color, lbl in [(0.1, "tomato", "d=0.1m"), (0.3, "orange", "d=0.3m"),
                                  (0.6, "steelblue", "d=0.6m"), (1.0, "green", "d=1.0m")]:
    if USE_FACING_COS:
        ax.plot(cos_range, a * cos_range / depth_fixed ** 2 + c * vel_c.mean() + b, color=color, label=lbl)
    else:
        ax.plot(cos_range, np.full_like(cos_range, a / depth_fixed ** 2 + c * vel_c.mean() + b), color=color, label=lbl)
ax.axhline(b, color="grey", linestyle="--", linewidth=0.8, label=f"floor b={b:.1f}")
ax.set_xlabel("facing_cos")
ax.set_ylabel("area (px²)")
ax.set_title("Area vs angle")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. Velocity effect (geometry term removed)
ax = axes[3]
resid_geom = areas_c - a * x_c  # geometry term removed, b left in on purpose (shows the floor)
ax.scatter(vel_c, resid_geom, s=8, alpha=0.4)
v_line = np.linspace(0, max(vel_c.max(), 1e-6), 300)
ax.plot(v_line, c * v_line + b, color="tomato", linewidth=1.5, label=f"c={c:.4f}")
ax.axhline(b, color="grey", linestyle="--", linewidth=0.8, label=f"floor b={b:.1f}")
ax.set_xlabel("velocity_px")
ax.set_ylabel("area residual (geometry removed)")
ax.set_title("Velocity effect")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "./fit_max_area.png"
plt.savefig(out_path, dpi=150)
plt.show()
print(f"\nPlot saved → {out_path}")
