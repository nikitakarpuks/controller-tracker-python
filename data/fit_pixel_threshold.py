"""
Fit pixel_threshold model:  threshold = a * facing_cos / depth_m² + b

Add data points below and run.  Prints fitted a, b and shows diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Data points ───────────────────────────────────────────────────────────────
# (depth_m, facing_cos, observed_pixel_threshold, description)
DATA = [
    (0.14784, 0.96268, 45, "in front of window"),
    (0.59929, 0.81538, 15, "far away"),
    (0.08782, 0.93871, 90, "very close"),
]

# ── Fit ───────────────────────────────────────────────────────────────────────
depths     = np.array([p[0] for p in DATA])
cos_vals   = np.array([p[1] for p in DATA])
thresholds = np.array([p[2] for p in DATA], dtype=float)
labels     = [p[3] for p in DATA]

x = cos_vals / depths ** 2          # physics regressor

A = np.column_stack([x, np.ones_like(x)])
(a, b), *_ = np.linalg.lstsq(A, thresholds, rcond=None)

print("=" * 52)
print(f"  pixel_threshold = {a:.5f} * cos / d² + {b:.5f}")
print("=" * 52)
print(f"\n{'Point':<28} {'measured':>9} {'predicted':>10} {'error':>7}")
print("-" * 56)
for depth, cos, thr, label in DATA:
    pred = a * cos / depth ** 2 + b
    print(f"  {label:<26} {thr:>9.1f} {pred:>10.1f} {pred - thr:>+7.1f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"pixel_threshold = {a:.4f} · cos/d²  +  {b:.4f}", fontsize=12)

# 1. Regression space: x = cos/d²  vs threshold
ax = axes[0]
x_line = np.linspace(0, x.max() * 1.1, 300)
ax.plot(x_line, a * x_line + b, color="steelblue", label="fit")
for xi, thr, label in zip(x, thresholds, labels):
    ax.scatter(xi, thr, zorder=5)
    ax.annotate(label, (xi, thr), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("cos / depth_m²")
ax.set_ylabel("pixel_threshold")
ax.set_title("Regression space")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Threshold vs depth (face-on LED, cos = 1.0)
ax = axes[1]
d_range = np.linspace(0.05, 0.8, 300)
ax.plot(d_range, a * 1.0 / d_range ** 2 + b, color="steelblue", label="cos=1.0 (face-on)")
ax.plot(d_range, a * 0.7 / d_range ** 2 + b, color="orange",    label="cos=0.7")
ax.plot(d_range, a * 0.4 / d_range ** 2 + b, color="tomato",    label="cos=0.4")
for depth, cos, thr, label in DATA:
    ax.scatter(depth, thr, zorder=5)
    ax.annotate(label, (depth, thr), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.axhline(b, color="grey", linestyle="--", linewidth=0.8, label=f"floor b={b:.1f}")
ax.set_xlabel("depth_m")
ax.set_ylabel("pixel_threshold")
ax.set_title("Threshold vs depth")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3. Threshold vs cos angle (at a few representative depths)
ax = axes[2]
cos_range = np.linspace(0.1, 1.0, 300)
for depth, color in [(0.1, "tomato"), (0.3, "orange"), (0.6, "steelblue"), (1.0, "green")]:
    ax.plot(cos_range, a * cos_range / depth ** 2 + b, color=color, label=f"d={depth}m")
for depth, cos, thr, label in DATA:
    ax.scatter(cos, thr, zorder=5)
    ax.annotate(label, (cos, thr), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.axhline(b, color="grey", linestyle="--", linewidth=0.8, label=f"floor b={b:.1f}")
ax.set_xlabel("facing_cos")
ax.set_ylabel("pixel_threshold")
ax.set_title("Threshold vs angle")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./data/fit_pixel_threshold.png", dpi=150)
plt.show()
print("\nPlot saved → data/fit_pixel_threshold.png")
