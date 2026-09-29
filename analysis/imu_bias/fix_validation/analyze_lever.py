"""analyze_lever.py -- tables + paired block-bootstrap CIs from results/ab_<rec>.csv (see ab_lever.py)."""
import numpy as np
import pandas as pd

import fv_common as C

R = C.HERE / "results"
df = pd.concat([pd.read_csv(R / f"ab_{n}.csv") for n in C.MODERATE], ignore_index=True)
LEV = ["a_zero", "b_factory_t (main.py today)", "c_bridge.t", "d_factory -R^T t", "e_old_design -Rb^T tb", "f_mean(c,d)"]
LOD = ["L0_OLD", "L1_drop2nd", "L2_scale_only", "L3_NEW"]
SETS = [("fixed", 0.1), ("fixed", 0.3), ("fixed", 1.0), ("real", None)]
out_lines = []


def P(s=""):
    print(s); out_lines.append(s)


def sel(ctrl, kind, T):
    s = df[(df.ctrl == ctrl) & (df.kind == kind)]
    return s[s["T"] == T] if T is not None else s


def label(kind, T):
    return f"T={T}s" if kind == "fixed" else "real gaps"


# ---------------------------------------------------------------- 1. lever A/B on the NEW loader (L3)
P("=" * 100)
P("1. LEVER A/B  (loader L3_NEW = recorded CSV, no 2nd factory correction, accel x0.980665).  Position error of the LED origin, mm")
P("   pooled over the 6 moderate recordings; no fitting anywhere -> no leakage.  median [p90]   (n gaps)")
rows = []
for ctrl in C.CTRLS:
    P(f"\n  {ctrl}")
    P(f"  {'candidate':32s}" + "".join(f"{label(k, T):>22s}" for k, T in SETS))
    for lev in LEV + ["naive"]:
        line = f"  {lev:32s}"
        for k, T in SETS:
            s = sel(ctrl, k, T)
            e = 1000 * (s["naive"] if lev == "naive" else s[f"L3_NEW|{lev}"])
            line += f"{np.median(e):8.1f} [{np.percentile(e, 90):7.1f}] "
            rows.append(dict(ctrl=ctrl, lever=lev, set=label(k, T), n=len(e), median_mm=np.median(e), p90_mm=np.percentile(e, 90), mean_mm=e.mean()))
        P(line)
    P(f"  {'n gaps':32s}" + "".join(f"{len(sel(ctrl, k, T)):>22d}" for k, T in SETS))
pd.DataFrame(rows).to_csv(R / "lever_ab_summary.csv", index=False)

# ---------------------------------------------------------------- paired differences with block bootstrap
P("\n" + "=" * 100)
P("2. PAIRED DIFFERENCES on L3_NEW (mm; negative = first is better), 95 % block-bootstrap CI (5 s blocks) of the MEAN and of the MEDIAN")
pairs = [("c_bridge.t", "d_factory -R^T t"), ("f_mean(c,d)", "d_factory -R^T t"), ("f_mean(c,d)", "c_bridge.t"),
         ("d_factory -R^T t", "b_factory_t (main.py today)"), ("c_bridge.t", "b_factory_t (main.py today)"),
         ("d_factory -R^T t", "a_zero"), ("e_old_design -Rb^T tb", "d_factory -R^T t"), ("e_old_design -Rb^T tb", "c_bridge.t")]
prow = []
for ctrl in C.CTRLS:
    P(f"\n  {ctrl}")
    for A, B in pairs:
        line = f"  {A[:22]:22s} - {B[:28]:28s}"
        for k, T in SETS:
            s = sel(ctrl, k, T)
            d = 1000 * (s[f"L3_NEW|{A}"] - s[f"L3_NEW|{B}"]).to_numpy(); tt = s["t_start"].to_numpy()
            lo, hi = C.block_bootstrap_ci(d, tt, stat=np.mean, n=1000)
            mlo, mhi = C.block_bootstrap_ci(d, tt, stat=np.median, n=1000)
            line += f" {d.mean():+7.2f}[{lo:+6.2f},{hi:+6.2f}]"
            prow.append(dict(ctrl=ctrl, A=A, B=B, set=label(k, T), mean_diff_mm=d.mean(), mean_lo=lo, mean_hi=hi, median_diff_mm=np.median(d), med_lo=mlo, med_hi=mhi))
        P(line)
    P("  columns: " + " | ".join(label(k, T) for k, T in SETS))
pd.DataFrame(prow).to_csv(R / "lever_paired.csv", index=False)

# ---------------------------------------------------------------- per recording consistency (which candidate wins)
P("\n" + "=" * 100)
P("3. PER-RECORDING median error at T=1.0 s (L3_NEW), mm")
for ctrl in C.CTRLS:
    P(f"\n  {ctrl}")
    P(f"  {'rec':14s}" + "".join(f"{l[:10]:>12s}" for l in LEV))
    for n in C.MODERATE:
        s = df[(df.rec == n) & (df.ctrl == ctrl) & (df.kind == "fixed") & (df["T"] == 1.0)]
        P(f"  {n:14s}" + "".join(f"{1000 * s['L3_NEW|' + l].median():12.1f}" for l in LEV))

# ---------------------------------------------------------------- sensitivity to the ~4 mm difference
P("\n" + "=" * 100)
P("4. SENSITIVITY: |error(c) - error(d)| (mm) and the physical size of the lever-term difference omega^2 * |dr| (m/s^2), L3_NEW")
for ctrl in C.CTRLS:
    cd = {}
    P(f"\n  {ctrl}: |c - d| lever vectors = {np.linalg.norm(C.lever_candidates(ctrl)['c_bridge.t'] - C.lever_candidates(ctrl)['d_factory -R^T t']) * 1000:.1f} mm ;"
      f" |c - e| = {np.linalg.norm(C.lever_candidates(ctrl)['c_bridge.t'] - C.lever_candidates(ctrl)['e_old_design -Rb^T tb']) * 1000:.1f} mm")
    for k, T in SETS:
        s = sel(ctrl, k, T)
        dd = 1000 * np.abs(s["L3_NEW|c_bridge.t"] - s["L3_NEW|d_factory -R^T t"])
        de = 1000 * np.abs(s["L3_NEW|c_bridge.t"] - s["L3_NEW|e_old_design -Rb^T tb"])
        om = s["omega_rms"].to_numpy()
        P(f"   {label(k, T):10s}  median|c-d| {np.median(dd):6.2f} mm  p90 {np.percentile(dd, 90):6.2f} | median|c-e| {np.median(de):6.2f} mm  p90 {np.percentile(de, 90):6.2f}"
          f" | median omega_rms {np.median(om):.2f} rad/s -> omega^2*|c-d| = {np.median(om) ** 2 * np.linalg.norm(C.lever_candidates(ctrl)['c_bridge.t'] - C.lever_candidates(ctrl)['d_factory -R^T t']):.3f} m/s^2")

# ---------------------------------------------------------------- 5. ATTRIBUTION (accel)
P("\n" + "=" * 100)
REC = "d_factory -R^T t"
P(f"5. ATTRIBUTION (accel position error of the LED origin, median mm; lever for the 'lever fixed' columns = {REC})")
P("   loader sub-changes:  L1 = drop the second mix+bias only,  L2 = accel x0.980665 only,  L3 = both")
arows = []
for ctrl in C.CTRLS:
    P(f"\n  {ctrl}")
    P(f"  {'variant':58s}" + "".join(f"{label(k, T):>12s}" for k, T in SETS))
    combos = [("OLD (L0 loader, factory lever)", "L0_OLD", "b_factory_t (main.py today)"),
              ("loader-only fix (L3, factory lever)", "L3_NEW", "b_factory_t (main.py today)"),
              ("   .. drop 2nd correction only (L1, factory lever)", "L1_drop2nd", "b_factory_t (main.py today)"),
              ("   .. accel scale only (L2, factory lever)", "L2_scale_only", "b_factory_t (main.py today)"),
              ("lever-only fix (L0 loader, recommended lever)", "L0_OLD", REC),
              ("BOTH (L3, recommended lever)", "L3_NEW", REC),
              ("   .. drop 2nd only + lever (L1)", "L1_drop2nd", REC),
              ("   .. scale only + lever (L2)", "L2_scale_only", REC),
              ("naive constant velocity", None, None)]
    for nm, lo, lv in combos:
        line = f"  {nm:58s}"
        for k, T in SETS:
            s = sel(ctrl, k, T)
            e = 1000 * (s["naive"] if lo is None else s[f"{lo}|{lv}"])
            line += f"{np.median(e):12.1f}"
            arows.append(dict(ctrl=ctrl, variant=nm.strip(), set=label(k, T), median_mm=np.median(e), mean_mm=e.mean(), p90_mm=np.percentile(e, 90), n=len(e)))
        P(line)
pd.DataFrame(arows).to_csv(R / "attribution_accel.csv", index=False)
(R / "lever_ab_report.txt").write_text("\n".join(out_lines))
