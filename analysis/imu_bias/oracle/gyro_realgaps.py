"""Real vision-loss gaps (no vision candidate for 0.15-1.2 s, from the raw vision_pose.csv): rotation prediction error of the gyro variants
vs mocap over the actual gap endpoints."""
import sys, csv
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from gyro_oracle import *
def load(name, ctrl, tau=0.5):
    z = np.load(OUT + f"gyro_terms_{name}_{ctrl}_tau{tau}.npz"); return {k: z[k] for k in z.files}
variants = ["V0_factory", "V2_const_bias+K", "V3_K_only", "V4_timevarying_bias+K(10s)"]
rows = []
for ctrl in CTRLS:
    K = np.load(OUT + f"K_pooled_{ctrl}.npy")
    for name in REC_NAMES:
        d = load(name, ctrl); idx = np.arange(len(d["t0"])); bK = solve(d, idx, K_fixed=K)[0]
        t_ = (d["t0"] - d["t0"][0]) / 1e9; Tc, Bw = [], []
        for s0 in np.arange(0, t_[-1] - 10 + 1e-9, 5):
            ii = np.flatnonzero((t_ >= s0) & (t_ < s0 + 10))
            if len(ii) >= 8: Bw.append(solve(d, ii, K_fixed=K)[0]); Tc.append(d["t0"][0] + (s0 + 5) * 1e9)
        Tc, Bw = np.array(Tc), np.array(Bw)
        rdir = rec_dir(name); mo = MocapOrientation(load_mocap_device(rdir, ctrl))
        t, gb, ab = load_imu(rdir, ctrl); gs = (DIAG_FLIP @ gb.T).T; gK = gs - gs @ K.T
        bt = np.stack([np.interp(t, Tc, Bw[:, i]) for i in range(3)], 1)
        streams = {"V0_factory": gs, "V2_const_bias+K": gK - bK, "V3_K_only": gK, "V4_timevarying_bias+K(10s)": gK - bt}
        ts = sorted({int(r[0]) for r in csv.reader(open(vision_csv(name))) if r[1] == ctrl and r[0].isdigit()})
        ts = np.array(ts); gaps = [(ts[i], ts[i+1]) for i in range(len(ts) - 1) if 0.15e9 <= ts[i+1] - ts[i] <= 1.2e9]
        for (a, b) in gaps:
            if not (mo.valid(np.array([a, b])).all() and a >= t[0] and b <= t[-1]): continue
            R0 = mo.R_world_imu(np.array([a])); R1 = mo.R_world_imu(np.array([b])); Rm = (R0[0].inv() * R1[0]).as_matrix()
            e = {}
            for v in variants:
                Rg = integrate_gyro_segment(t, streams[v], int(a), int(b))
                e[v] = np.degrees(np.linalg.norm(Rotation.from_matrix(Rm.T @ Rg).as_rotvec())) if Rg is not None else np.nan
            rows.append(dict(rec=name, ctrl=ctrl, gap_s=(b - a) / 1e9, t_start_s=(a - t[0]) / 1e9, **e))
with open(OUT + "gyro_realgaps.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} real gaps (0.15-1.2 s)")
for lo, hi in ((0.15, 0.3), (0.3, 0.6), (0.6, 1.2), (0.15, 1.2)):
    sub = [r for r in rows if lo <= r["gap_s"] < hi + 1e-9]
    if not sub: continue
    print(f"gap {lo}-{hi}s n={len(sub):3d}: " + "  ".join(f"{v.split('_')[0]}{v.split('_',1)[1][:14]}: med {np.nanmedian([r[v] for r in sub]):6.2f} mean {np.nanmean([r[v] for r in sub]):6.2f}" for v in variants))
