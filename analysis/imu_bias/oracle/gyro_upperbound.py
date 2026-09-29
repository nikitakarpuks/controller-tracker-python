"""Upper bound: replace the gyro stream with ORACLE-corrected variants and score rotation prediction against mocap over gaps.
error = angle( Rm^T Rg ), Rm = mocap relative rotation (IMU frame), Rg = integrated (corrected) gyro in the sensor frame.
NOTE: the score has a mocap floor (orientation noise ~0.17 deg/sample -> ~0.24 deg for a difference) that no correction can remove."""
import sys, csv
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from gyro_oracle import *
def load(name, ctrl, tau=0.5):
    z = np.load(OUT + f"gyro_terms_{name}_{ctrl}_tau{tau}.npz"); return {k: z[k] for k in z.files}
DUR = [0.022, 0.044, 0.1, 0.3, 1.0]; NST = 250
rng = np.random.default_rng(11)
results = []   # rows: rec, ctrl, variant, T, err_deg (list)
variants = ["V0_factory", "V1_const_bias_only", "V2_const_bias+K", "V3_K_only", "V4_timevarying_bias+K(10s)", "V5_single_bias+K_all_recs"]
for ctrl in CTRLS:
    K = np.load(OUT + f"K_pooled_{ctrl}.npy")
    # per-recording constants
    bK = {}; b_bo = {}; bw = {}
    for name in REC_NAMES:
        d = load(name, ctrl); idx = np.arange(len(d["t0"]))
        b_bo[name] = solve(d, idx, with_K=False)[0]
        bK[name] = solve(d, idx, K_fixed=K)[0]
        # 10 s windows, stride 5 s, K fixed
        t = (d["t0"] - d["t0"][0]) / 1e9; Tc, Bw = [], []
        for s0 in np.arange(0, t[-1] - 10 + 1e-9, 5):
            ii = np.flatnonzero((t >= s0) & (t < s0 + 10))
            if len(ii) < 8: continue
            Bw.append(solve(d, ii, K_fixed=K)[0]); Tc.append(d["t0"][0] + (s0 + 5) * 1e9)
        bw[name] = (np.array(Tc), np.array(Bw))
    b_all = np.median(np.array([bK[n] for n in REC_NAMES]), 0)
    print(f"{ctrl}: single bias (median over recs, K pooled) = {b_all.round(4)} rad/s", flush=True)
    for name in REC_NAMES:
        rdir = rec_dir(name); mo = MocapOrientation(load_mocap_device(rdir, ctrl))
        t, gb, ab = load_imu(rdir, ctrl); gs = (DIAG_FLIP @ gb.T).T
        gK = gs - gs @ K.T
        Tc, Bw = bw[name]
        bt = np.stack([np.interp(t, Tc, Bw[:, i]) for i in range(3)], 1)
        streams = {"V0_factory": gs, "V1_const_bias_only": gs - b_bo[name], "V2_const_bias+K": gK - bK[name], "V3_K_only": gK,
                   "V4_timevarying_bias+K(10s)": gK - bt, "V5_single_bias+K_all_recs": gK - b_all}
        for T in DUR:
            dtn = int(T * 1e9)
            cand = np.arange(t[0] + 0.5e9, t[-1] - dtn - 0.5e9, 1e7).astype(np.int64)
            ok = mo.valid(cand) & mo.valid(cand + dtn); cand = cand[ok]
            starts = rng.choice(cand, min(NST, len(cand)), replace=False)
            R0 = mo.R_world_imu(starts); R1 = mo.R_world_imu(starts + dtn)
            errs = {v: [] for v in variants}
            for i, s0 in enumerate(starts):
                Rm = (R0[i].inv() * R1[i]).as_matrix()
                for v in variants:
                    Rg = integrate_gyro_segment(t, streams[v], int(s0), int(s0) + dtn)
                    errs[v].append(np.degrees(np.linalg.norm(Rotation.from_matrix(Rm.T @ Rg).as_rotvec())))
            om = np.degrees(np.linalg.norm(Rotation.from_matrix(Rm).as_rotvec()))
            for v in variants: results.append((name, ctrl, v, T, np.array(errs[v])))
        print(f"  {name} done", flush=True)
import pickle; pickle.dump(results, open(OUT + "gyro_upperbound_raw.pkl", "wb"))
rows = []
for v in variants:
    for T in DUR:
        for scope, names in (("all", REC_NAMES), ("static_dark", ["static_dark"]), ("moderate6", ["static_dark","walk_dark","static_easy","static_medium","walk_easy","walk_medium"])):
            e = np.concatenate([r[4] for r in results if r[2] == v and r[3] == T and r[0] in names])
            rows.append(dict(variant=v, T_s=T, scope=scope, n=len(e), median_deg=np.median(e), mean_deg=e.mean(), p90_deg=np.percentile(e, 90)))
with open(OUT + "gyro_upperbound.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nmedian rotation prediction error (deg), scope=all recordings, both controllers:")
print(f"{'variant':30s}" + "".join(f"T={T:<7}" for T in DUR))
for v in variants:
    print(f"{v:30s}" + "".join(f"{[r for r in rows if r['variant']==v and r['T_s']==T and r['scope']=='all'][0]['median_deg']:<9.3f}" for T in DUR))
print("\nmean error (deg):")
for v in variants:
    print(f"{v:30s}" + "".join(f"{[r for r in rows if r['variant']==v and r['T_s']==T and r['scope']=='all'][0]['mean_deg']:<9.3f}" for T in DUR))
