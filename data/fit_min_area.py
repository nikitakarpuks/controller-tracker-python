"""
Fit min_area model:  min_area = a * facing_cos / depth_m² + b

Lower-bound fit: the curve is placed so that LOWER_PERCENTILE % of
non-outlier blobs fall ABOVE it — i.e. no real LED blob gets filtered out.
Outliers (tiny sub-pixel / edge-cropped blobs) are excluded before fitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH     = "./calib_warm.csv"
LOWER_PCTILE = 0.5   # % of clean data allowed to fall below (i.e. 95% must be above)

# ── Load & prepare ────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")

depths   = df["depth_m"].values
cos_vals = df["facing_cos"].values
areas    = df["area"].values
x = cos_vals / depths ** 2

# ── Remove outliers (tiny edge-cropped blobs) via IQR on raw areas ───────────
q1, q3   = np.percentile(areas, [25, 75])
iqr      = q3 - q1
lower_cut = q1 - 8.0 * iqr
clean     = areas >= max(lower_cut, 0.0)

x_c, areas_c = x[clean], areas[clean]
print(f"After outlier removal: {clean.sum()} / {len(areas)} rows kept "
      f"(cut at area < {max(lower_cut, 0.0):.2f} px²)")

# ── Fit slope 'a' with least squares on clean data ───────────────────────────
A = np.column_stack([x_c, np.ones_like(x_c)])
(a, _), *_ = np.linalg.lstsq(A, areas_c, rcond=None)

# ── Set 'b' so (100 - LOWER_PCTILE)% of clean data lies ABOVE the curve ─────
residuals_c = areas_c - a * x_c
b = float(np.percentile(residuals_c, LOWER_PCTILE))

above = (areas_c >= a * x_c + b).mean() * 100
print("=" * 56)
print(f"  min_area = {a:.5f} * cos / d²  +  {b:.5f}")
print(f"  {above:.1f}% of clean blobs fall above the curve")
print("=" * 56)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(
    f"min_area = {a:.4f} · cos/d²  +  {b:.4f}   "
    f"({above:.0f}% above, {100 - LOWER_PCTILE}th-pctile bound, n={clean.sum()})",
    fontsize=11,
)

# 1. Regression space
ax = axes[0]
x_line = np.linspace(0, x_c.max() * 1.05, 300)
ax.scatter(x[~clean], areas[~clean], s=12, color="lightgrey", alpha=0.6, label="outliers (excluded)")
ax.scatter(x_c, areas_c,            s=8,  alpha=0.4,          label="clean observations")
ax.plot(x_line, a * x_line + b, color="steelblue", linewidth=1.5, label=f"{100 - LOWER_PCTILE}th pctile bound")
ax.set_xlabel("cos / depth_m²")
ax.set_ylabel("area (px²)")
ax.set_title("Regression space")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Area vs depth
ax = axes[1]
d_range = np.linspace(0.05, 1.0, 300)
ax.scatter(depths[~clean], areas[~clean], s=12, color="lightgrey", alpha=0.6)
ax.scatter(depths[clean],  areas[clean],  s=8,  alpha=0.4)
for cos_fixed, color, lbl in [(1.0, "steelblue", "cos=1.0"), (0.7, "orange", "cos=0.7"), (0.4, "tomato", "cos=0.4")]:
    ax.plot(d_range, a * cos_fixed / d_range ** 2 + b, color=color, label=lbl)
ax.axhline(b, color="grey", linestyle="--", linewidth=0.8, label=f"floor b={b:.1f}")
ax.set_xlabel("depth_m")
ax.set_ylabel("area (px²)")
ax.set_title("Area vs depth")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3. Area vs cos
ax = axes[2]
cos_range = np.linspace(0.1, 1.0, 300)
ax.scatter(cos_vals[~clean], areas[~clean], s=12, color="lightgrey", alpha=0.6)
ax.scatter(cos_vals[clean],  areas[clean],  s=8,  alpha=0.4)
for depth_fixed, color, lbl in [(0.1, "tomato", "d=0.1m"), (0.3, "orange", "d=0.3m"),
                                  (0.6, "steelblue", "d=0.6m"), (1.0, "green", "d=1.0m")]:
    ax.plot(cos_range, a * cos_range / depth_fixed ** 2 + b, color=color, label=lbl)
ax.axhline(b, color="grey", linestyle="--", linewidth=0.8, label=f"floor b={b:.1f}")
ax.set_xlabel("facing_cos")
ax.set_ylabel("area (px²)")
ax.set_title("Area vs angle")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "./fit_min_area.png"
plt.savefig(out_path, dpi=150)
plt.show()
print(f"\nPlot saved → {out_path}")
