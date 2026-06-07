import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_CSV           = r".\calib_thresholds_raw.csv"
OUTPUT_CSV          = r".\calib_thresholds_raw_filtered.csv"

N_BINS              = 50   # output rows (depth levels)
OUTLIER_K           = 2.0  # reject > this many MADs from local median
ROLLING_WINDOW      = 50   # window size (depth-sorted rows) for outlier detection
MIN_SAMPLES_PER_BIN = 5    # drop bins with fewer surviving samples
# ─────────────────────────────────────────────────────────────────────────────

def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def inv_depth(d, a, b):       # a/d + b
    return a / d + b

def inv_depth_sq(d, a, b):    # a/d² + b
    return a / d**2 + b

def fit_both(depths, values, label):
    """Fit a/d+b and a/d²+b, print R² for both, return best (params, model_fn, name)."""
    results = []
    for fn, name, p0 in [(inv_depth, "a/d+b", [1.0, 0.0]),
                          (inv_depth_sq, "a/d²+b", [0.1, 0.0])]:
        try:
            popt, _ = curve_fit(fn, depths, values, p0=p0, maxfev=5000)
            r2 = _r2(values, fn(depths, *popt))
            results.append((popt, fn, name, r2))
        except Exception as e:
            print(f"  {label} {name} fit failed: {e}")
    if not results:
        return None
    results.sort(key=lambda x: -x[3])
    for popt, fn, name, r2 in results:
        print(f"  {label:22s} {name:8s}  a={popt[0]:9.4f}  b={popt[1]:8.4f}  R²={r2:.4f}"
              + ("  ← best" if fn is results[0][1] else ""))
    return results[0]   # (popt, fn, name, r2) for best

# ── 1. Load & clean ───────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows")

df = df.dropna(subset=["depth_m", "pixel_threshold", "required_threshold"])
df = df[df["depth_m"] > 0]
print(f"After cleanup: {len(df)} rows  |  "
      f"depth [{df['depth_m'].min():.3f}, {df['depth_m'].max():.3f}] m")

# ── 2. Sort by depth ──────────────────────────────────────────────────────────
df = df.sort_values("depth_m").reset_index(drop=True)

# ── 3. Rolling-window outlier removal ────────────────────────────────────────
threshold_cols = [c for c in ["pixel_threshold", "required_threshold", "max_area"]
                  if c in df.columns]

outlier_mask = pd.Series(False, index=df.index)
for col in threshold_cols:
    vals = df[col].astype(float)
    local_med = vals.rolling(ROLLING_WINDOW, center=True, min_periods=1).median()
    local_mad = (vals - local_med).abs().rolling(ROLLING_WINDOW, center=True, min_periods=1).median()
    fallback   = vals.std() * 0.6745 or 1.0
    local_mad  = local_mad.replace(0, np.nan).fillna(fallback)
    outlier_mask |= (vals - local_med).abs() > OUTLIER_K * local_mad

outlier_df = df[outlier_mask].copy()
clean_df   = df[~outlier_mask].reset_index(drop=True)
print(f"Outlier removal: dropped {len(outlier_df)} → {len(clean_df)} remaining")

# ── 4. Equal-count quantile bins → median per bin ────────────────────────────
clean_df["_bin"] = pd.qcut(clean_df["depth_m"], q=N_BINS, duplicates="drop")

agg = (
    clean_df.groupby("_bin", observed=True)
    .agg(
        depth_m            = ("depth_m",            "median"),
        pixel_threshold    = ("pixel_threshold",    "median"),
        required_threshold = ("required_threshold", "median"),
        max_area           = ("max_area",           "median"),
        count              = ("depth_m",            "count"),
        pixel_thr_std      = ("pixel_threshold",    "std"),
        req_thr_std        = ("required_threshold", "std"),
        max_area_std       = ("max_area",           "std"),
    )
    .reset_index(drop=True)
    .sort_values("depth_m")
)

agg = agg[agg["count"] >= MIN_SAMPLES_PER_BIN].reset_index(drop=True)

agg["pixel_threshold"]    = agg["pixel_threshold"].round().astype(int)
agg["required_threshold"] = agg["required_threshold"].round().astype(int)
agg["max_area"]           = agg["max_area"].round(1)
agg["pixel_thr_std"]      = agg["pixel_thr_std"].round(2)
agg["req_thr_std"]        = agg["req_thr_std"].round(2)
agg["depth_m"]            = agg["depth_m"].round(4)

print(f"\nOutput: {len(agg)} bins")
print(agg[["depth_m", "pixel_threshold", "required_threshold",
           "max_area", "count", "pixel_thr_std"]].to_string(index=False))

# ── 5. Save filtered CSV ──────────────────────────────────────────────────────
out_cols = ["depth_m", "pixel_threshold", "required_threshold", "max_area",
            "count", "pixel_thr_std", "req_thr_std"]
agg[out_cols].to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}")

# ── 6. Fit both models for each column ───────────────────────────────────────
depths = agg["depth_m"].values.astype(float)

print("\n── Model fits (on bin medians) ──────────────────────────────────────────")
fits = {}
for col in ["pixel_threshold", "required_threshold", "max_area"]:
    if col not in agg.columns:
        continue
    vals = agg[col].values.astype(float)
    best = fit_both(depths, vals, col)
    fits[col] = best   # (popt, fn, name, r2)

# ── 7. Print config-ready block ───────────────────────────────────────────────
print("\n── Config-ready coefficients (best model per column) ────────────────────")
print("pose_guided_thresholds:")
for col in ["pixel_threshold", "required_threshold", "max_area"]:
    if col not in fits or fits[col] is None:
        continue
    popt, fn, name, r2 = fits[col]
    model_tag = "inv_depth" if fn is inv_depth else "inv_depth_sq"
    print(f"  {col}:")
    print(f"    model: {model_tag!r}   # {name}  R²={r2:.4f}")
    print(f"    a: {popt[0]:.6f}")
    print(f"    b: {popt[1]:.6f}")

# ── 8. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle("Calibration data: depth vs detection thresholds", fontsize=13)

raw_kw  = dict(s=4,  alpha=0.2,  color="steelblue", label="raw (clean)")
out_kw  = dict(s=10, alpha=0.6,  color="tomato",    marker="x", label="outlier")
fit1_kw = dict(color="darkorange", lw=1.8, ls="--", label="a/d+b")
fit2_kw = dict(color="green",      lw=1.8, ls=":",  label="a/d²+b")

depth_raw  = df["depth_m"].values
depth_fine = np.linspace(df["depth_m"].min(), df["depth_m"].max(), 300)

col_specs = [
    (axes[0, 0], "pixel_threshold",    "pixel_thr_std",  "pixel_threshold"),
    (axes[0, 1], "required_threshold", "req_thr_std",    "required_threshold"),
    (axes[0, 2], "max_area",           "max_area_std",   "max_area"),
]

for ax, col, std_col, title in col_specs:
    ax.scatter(depth_raw[~outlier_mask], df.loc[~outlier_mask, col], **raw_kw)
    if len(outlier_df):
        ax.scatter(outlier_df["depth_m"], outlier_df[col], **out_kw)
    ax.errorbar(agg["depth_m"], agg[col], yerr=agg[std_col],
                fmt="o", color="navy", zorder=5,
                markersize=6, capsize=3, label="bin median ± std")
    vals = agg[col].values.astype(float)
    for fn, kw in [(inv_depth, fit1_kw), (inv_depth_sq, fit2_kw)]:
        try:
            popt, _ = curve_fit(fn, depths, vals, p0=[1.0, 0.0], maxfev=5000)
            r2 = _r2(vals, fn(depths, *popt))
            kw_r2 = dict(kw, label=f"{kw['label']}  R²={r2:.3f}")
            ax.plot(depth_fine, fn(depth_fine, *popt), **kw_r2)
        except Exception:
            pass
    ax.set_xlabel("depth (m)")
    ax.set_ylabel(col)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Bottom row: 1/depth and 1/depth² linearisation plots
for ax, col, x_fn, x_label, title_suffix in [
    (axes[1, 0], "pixel_threshold",    lambda d: 1/d,    "1/depth (m⁻¹)",  "pixel_threshold — 1/d space"),
    (axes[1, 1], "required_threshold", lambda d: 1/d,    "1/depth (m⁻¹)",  "required_threshold — 1/d space"),
    (axes[1, 2], "max_area",           lambda d: 1/d**2, "1/depth² (m⁻²)", "max_area — 1/d² space"),
]:
    x_raw  = x_fn(depth_raw)
    x_bins = x_fn(depths)
    x_fine = x_fn(depth_fine)

    ax.scatter(x_raw[~outlier_mask], df.loc[~outlier_mask, col], **raw_kw)
    if len(outlier_df):
        ax.scatter(x_fn(outlier_df["depth_m"].values), outlier_df[col], **out_kw)
    ax.scatter(x_bins, agg[col], s=60, color="navy", zorder=5, label="bin median")

    vals = agg[col].values.astype(float)
    try:
        p = np.polyfit(x_bins, vals, 1)
        y_pred = np.polyval(p, x_bins)
        r2 = _r2(vals, y_pred)
        ax.plot(x_fine, np.polyval(p, x_fine), color="darkorange", lw=1.8, ls="--",
                label=f"linear fit  R²={r2:.3f}")
    except Exception:
        pass

    ax.set_xlabel(x_label)
    ax.set_ylabel(col)
    ax.set_title(title_suffix)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
