"""run_experiments.py -- Exp2 (convergence/steady-state), Exp3 (prediction benefit), Exp4 (sensitivity), Exp5 (design choices).
Usage: python3 run_experiments.py <exp2|exp4|exp5|all> [n_workers]
All results -> CSV next to this file; example bias traces -> traces/*.npz."""
import sys, time, json, itertools, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from simlib import *
from estlib import *
from estlib import _lever_terms

OUT = Path(__file__).parent; (OUT / "traces").mkdir(exist_ok=True)
T_SS = 60.0            # steady-state metrics from this time (s since first IMU sample)
HORIZONS = [0.022, 0.044, 0.088, 0.25, 0.5, 1.0]

SCEN = {
    "factory": dict(b_g0=np.array([1e-4, -8e-5, 1.2e-4]), b_a0=np.array([0.01, -0.008, 0.006]), rw_g=1e-5, rw_a=1e-3),
    "large":   dict(b_g0=np.array([0.011, -0.008, 0.014]), b_a0=np.array([0.2, -0.15, 0.1]), rw_g=1e-5, rw_a=1e-3),
    "drift":   dict(b_g0=np.array([1e-4, -8e-5, 1.2e-4]), b_a0=np.array([0.01, -0.008, 0.006]), rw_g=1e-5, rw_a=1e-3,
                    warm_g=np.array([0.004, -0.003, 0.0035]), warm_a=np.array([0.03, -0.02, 0.025]), warm_tau=40.0),
}


def node_series(d, arr):
    tv = d.t_v_s
    return np.stack([np.interp(tv, d.t_imu_s, arr[:, i]) for i in range(3)], 1)


def series_metrics(tv, est, truth, name, extra):
    m = tv > T_SS
    e = est[m] - truth[m]; z = truth[m]
    rms = float(np.sqrt((e ** 2).mean())); rms0 = float(np.sqrt((z ** 2).mean()))
    err = np.linalg.norm(est - truth, axis=1); mag = np.linalg.norm(truth, axis=1)
    thr = np.maximum(0.3 * mag, 1e-9)
    ok = err < thr
    t_conv = np.nan
    for i in range(len(tv)):
        if ok[i:].mean() > 0.95 and (tv[-1] - tv[i]) > 10: t_conv = float(tv[i] - tv[0]); break
    return dict(estimator=name, rms=rms, rms_zero=rms0, ratio_vs_zero=rms / max(rms0, 1e-12), mean_err=float(np.linalg.norm(e.mean(0))),
                t_conv_30pct=t_conv, **extra)


def predict_benefit(d, bg_ser, ba_ser, tag, t_starts, out_rows, extra):
    """Dead-reckoning prediction error from TRUE state at t_s with (zero | estimated | true) bias."""
    t_imu = d.t_imu_ns; tr = d.truth; T0 = d.T0_ns
    lv_all = _lever_terms(d.t_imu_s, d.gyro, d.lever_est)
    tv = d.t_v_s
    for h in HORIZONS:
        for opt in ("zero", "est", "true"):
            re, pe = [], []
            for ts in t_starts:
                k = max(int(np.searchsorted(tv, ts, "right")) - 1, 0)
                if opt == "zero": bg = ba = np.zeros(3)
                elif opt == "est": bg, ba = bg_ser[k], ba_ser[k]
                else:
                    i = int(np.searchsorted(d.t_imu_s, ts)); bg, ba = d.b_g_true[i], d.b_a_true[i]
                ns0 = int(round(ts * 1e9)) + T0; ns1 = ns0 + int(round(h * 1e9))
                if ns1 > t_imu[-1]: continue
                times, P, dt, a = gyro_prefix(t_imu, d.gyro, ns0, ns1, b=bg)
                R0 = tr.R(ts)[0]; Ri = np.einsum("ij,njk->nik", R0, P)
                f = np.stack([np.interp(times, t_imu.astype(float), d.accel[:, j]) for j in range(3)], 1) - ba \
                    - np.stack([np.interp(times, t_imu.astype(float), lv_all[:, j]) for j in range(3)], 1)
                aw = np.einsum("nij,nj->ni", Ri, f) + d.g_est
                ts_s = times / 1e9
                v = _cum(ts_s, aw) + tr.v(ts)[0]; p = _cum(ts_s, v)
                t1 = ts + h
                re.append(np.degrees(np.linalg.norm(Log(Ri[-1].T @ tr.R(t1)[0]))))
                pe.append(np.linalg.norm(p[-1] - (tr.p(t1)[0] - tr.p(ts)[0])) * 1000)
            out_rows.append(dict(estimator=tag, opt=opt, horizon_s=h, rot_err_deg_median=float(np.median(re)), rot_err_deg_p90=float(np.percentile(re, 90)),
                                 pos_err_mm_median=float(np.median(pe)), pos_err_mm_p90=float(np.percentile(pe, 90)), n=len(re), **extra))


def _cum(t, x):
    o = np.zeros_like(x); o[1:] = np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(t)[:, None], axis=0); return o


def run_estimators(d, full=True):
    """Returns dict name -> (b_g (K,3), b_a (K,3) or None)."""
    iv = Intervals(d); K = len(d.t_v_ns); res = {}
    for mode, tau in (("naive_ema", 10), ("naive_ema", 30), ("dt_weighted", 10), ("dt_weighted", 30), ("dt_weighted", 60), ("median", 30)):
        res[f"A_{mode}_tau{tau}"] = (run_A_gyro(d, iv, mode, tau), None)
    if full:
        for W in (30, 60):
            b, _ = run_C_gyro(d, iv, W=W, every=15); res[f"C_gyro_W{W}"] = (b, None)
    for q in (3e-6, 1e-5, 3e-5):
        r = run_B(d, q_bg=q, q_ba=1e-3); res[f"B_q{q:g}"] = (r["b_g"], r["b_a"])
    bgA = res["A_dt_weighted_tau30"][0]
    for mode in ("naive_ema", "dt_weighted"):
        res[f"A_accel_{mode}"] = (None, run_A_accel(d, iv, bgA, mode, 30.0))
    if full:
        bc, _ = run_C_gyro(d, iv, W=30, every=15)
        ba, _ = run_C_accel(d, bc, W=30, every=50); res["C_accel_W30"] = (None, ba)
    return res


def exp2_worker(args):
    rec, dev, sname, seed = args
    d = simulate(rec, dev, Scenario(name=sname, seed=seed, **SCEN[sname]))
    tv = d.t_v_s; bgT, baT = node_series(d, d.b_g_true), node_series(d, d.b_a_true)
    res = run_estimators(d); rows = []; ben = []
    base = dict(rec=rec, dev=dev, scenario=sname, seed=seed, n_nodes=len(tv))
    rows.append(dict(estimator="zero", rms=float(np.sqrt((bgT[tv > T_SS] ** 2).mean())), rms_zero=float(np.sqrt((bgT[tv > T_SS] ** 2).mean())), ratio_vs_zero=1.0, kind="gyro", **base))
    rows.append(dict(estimator="zero", rms=float(np.sqrt((baT[tv > T_SS] ** 2).mean())), rms_zero=float(np.sqrt((baT[tv > T_SS] ** 2).mean())), ratio_vs_zero=1.0, kind="accel", **base))
    for name, (bg, ba) in res.items():
        if bg is not None: rows.append({**series_metrics(tv, bg, bgT, name, dict(kind="gyro")), **base})
        if ba is not None: rows.append({**series_metrics(tv, ba, baT, name, dict(kind="accel")), **base})
    if seed == 0:
        np.savez_compressed(OUT / "traces" / f"{rec}_{dev}_{sname}.npz", tv=tv, bgT=bgT, baT=baT,
                            **{f"{n}__g": v[0] for n, v in res.items() if v[0] is not None}, **{f"{n}__a": v[1] for n, v in res.items() if v[1] is not None})
    # ---- Exp3 benefit: A_dtw(tau30), C_gyro W30 (gyro only), B (both)
    t_starts = np.arange(T_SS, tv[-1] - 1.5, 0.4)
    zero = np.zeros((len(tv), 3))
    for tag, bg, ba in (("A_dtw30+accel0", res["A_dt_weighted_tau30"][0], zero), ("C_gyro_W30+accel0", res["C_gyro_W30"][0], zero),
                        ("B_q1e-05", res["B_q1e-05"][0], res["B_q1e-05"][1])):
        predict_benefit(d, bg, ba, tag, t_starts, ben, base)
    return rows, ben


# ------------------------------------------------------------------ Exp4: sensitivity (one failure at a time)
SENS = [("none", {}),
        ("timing 1ms", dict(timing_ms=1.0)), ("timing 3ms", dict(timing_ms=3.0)), ("timing 8ms", dict(timing_ms=8.0)),
        ("headset ignored", dict(headset_mode="ignored")), ("headset vio drift 1e-3", dict(headset_mode="vio_drift", headset_drift=1e-3)),
        ("headset vio drift 3e-3", dict(headset_mode="vio_drift", headset_drift=3e-3)),
        ("lever +20%", dict(lever_err=0.2)), ("lever -50%", dict(lever_err=-0.5)), ("lever ignored (-100%)", dict(lever_err=-1.0)),
        ("gravity tilt 0.5deg", dict(grav_tilt_deg=0.5)), ("gravity tilt 1deg", dict(grav_tilt_deg=1.0)), ("gravity tilt 2deg", dict(grav_tilt_deg=2.0)),
        ("gravity mag +1%", dict(grav_mag_err=0.01)),
        ("axis misalign 0.5deg", dict(axis_mis_deg=0.5)), ("axis misalign 1deg", dict(axis_mis_deg=1.0)), ("axis misalign 2deg", dict(axis_mis_deg=2.0)),
        ("gyro scale 0.1%", dict(scale_g=1e-3)), ("gyro scale 0.5%", dict(scale_g=5e-3)), ("accel scale 0.3%", dict(scale_a=3e-3)),
        ("vision outliers 5%", dict(outlier_frac=0.05)), ("identity swap 3s", dict(swap_window=(70.0, 73.0))),
        ("corr vision err 0.5deg/2mm", dict(corr_rot_deg=0.5, corr_pos_mm=2.0)), ("corr vision err 1deg/4mm", dict(corr_rot_deg=1.0, corr_pos_mm=4.0)),
        ("vision noise x2", dict(sig_rot_deg=0.8, sig_pos_mm=8.0)),
        ("dt jitter 0.5ms", dict(jitter_ms=0.5)), ("dt jitter 1ms", dict(jitter_ms=1.0))]


def exp4_worker(args):
    rec, dev, (label, kw), seed = args
    d = simulate(rec, dev, Scenario(name=label, seed=seed, **kw))          # TRUE bias = 0 -> any estimate is an artifact
    if kw.get("headset_mode") == "ignored":
        pass
    tv = d.t_v_s; iv = Intervals(d); rows = []
    base = dict(rec=rec, dev=dev, case=label, seed=seed)
    out = {}
    out["A_dtw30"] = (run_A_gyro(d, iv, "dt_weighted", 30.0), None)
    b, _ = run_C_gyro(d, iv, W=30, every=25); out["C_gyro_W30"] = (b, None)
    r = run_B(d, q_bg=1e-5, q_ba=1e-3); out["B"] = (r["b_g"], r["b_a"])
    for name, (bg, ba) in out.items():
        m = tv > T_SS
        if bg is not None:
            rows.append(dict(estimator=name, kind="gyro", mean_artifact=float(np.linalg.norm(bg[m].mean(0))), rms_artifact=float(np.sqrt((bg[m] ** 2).sum(1).mean())), **base))
        if ba is not None:
            rows.append(dict(estimator=name, kind="accel", mean_artifact=float(np.linalg.norm(ba[m].mean(0))), rms_artifact=float(np.sqrt((ba[m] ** 2).sum(1).mean())), **base))
    return rows


# ------------------------------------------------------------------ Exp5: design choices for B and A
def exp5_worker(args):
    rec, dev, sname, seed, extra_kw = args
    kw = {**SCEN[sname], **extra_kw.get("scen", {})}
    d = simulate(rec, dev, Scenario(name=sname, seed=seed, **kw)); tv = d.t_v_s
    bgT, baT = node_series(d, d.b_g_true), node_series(d, d.b_a_true); rows = []
    base = dict(rec=rec, dev=dev, scenario=sname + extra_kw.get("tag", ""), seed=seed)
    for q_bg, infl, sig_bg0, q_ba in itertools.product((1e-6, 3e-6, 1e-5, 3e-5, 1e-4), (1.0, 1.5, 2.5), (0.02,), (1e-4, 1e-3, 1e-2)):
        r = run_B(d, q_bg=q_bg, q_ba=q_ba, meas_infl=infl, sig_bg0=sig_bg0)
        rows.append({**series_metrics(tv, r["b_g"], bgT, "B", dict(kind="gyro", q_bg=q_bg, meas_infl=infl, q_ba=q_ba, n_rej=r["n_rej"])), **base})
        rows.append({**series_metrics(tv, r["b_a"], baT, "B", dict(kind="accel", q_bg=q_bg, meas_infl=infl, q_ba=q_ba, n_rej=r["n_rej"])), **base})
    iv = Intervals(d)
    for gate in (0.02, 0.05, 0.12, 0.5):
        for tau in (10, 30, 60):
            rows.append({**series_metrics(tv, run_A_gyro(d, iv, "dt_weighted", tau, gate=gate), bgT, "A_dtw", dict(kind="gyro", gate=gate, tau=tau)), **base})
    return rows


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    t0 = time.time()
    if which in ("exp2", "all"):
        jobs = [("static_dark", "right", s, sd) for s in SCEN for sd in range(3)] + [("static_dark", "left", s, sd) for s in ("factory", "large") for sd in range(2)] \
             + [("walk_medium", "right", s, 0) for s in ("factory", "large")]
        R, Bn = [], []
        with ProcessPoolExecutor(nw) as ex:
            for rows, ben in ex.map(exp2_worker, jobs):
                R += rows; Bn += ben
        pd.DataFrame(R).to_csv(OUT / "exp2_convergence.csv", index=False); pd.DataFrame(Bn).to_csv(OUT / "exp3_benefit.csv", index=False)
        print("exp2/3 done", time.time() - t0)
    if which in ("exp4", "all"):
        jobs = [("static_dark", "right", c, sd) for c in SENS for sd in range(2)]
        R = []
        with ProcessPoolExecutor(nw) as ex:
            for rows in ex.map(exp4_worker, jobs): R += rows
        pd.DataFrame(R).to_csv(OUT / "exp4_sensitivity.csv", index=False); print("exp4 done", time.time() - t0)
    if which in ("exp5", "all"):
        jobs = [("static_dark", "right", s, sd, {}) for s in ("factory", "large", "drift") for sd in range(2)] \
             + [("static_dark", "right", "factory", sd, dict(scen=dict(corr_rot_deg=0.5, corr_pos_mm=2.0), tag="+corr_vision")) for sd in range(2)]
        R = []
        with ProcessPoolExecutor(nw) as ex:
            for rows in ex.map(exp5_worker, jobs): R += rows
        pd.DataFrame(R).to_csv(OUT / "exp5_design.csv", index=False); print("exp5 done", time.time() - t0)


if __name__ == "__main__":
    main()
