"""Gyro bias oracle from mocap, first-order exact-Jacobian model.
Model (sensor frame, after project's factory T=0 correction):  gyro_meas = (I+K) w_true + b.
Interval k (tau seconds, stride tau): r0_k = Log(Rm_k^T Rg_k(0)),  Rg_k = exact midpoint integration of the measured gyro,
 Rm_k = mocap R(t0)^T R(t1) (mocap IMU frame == sensor frame, verified by gate_mocap_gyro.py).
Perturbing corrected rates by delta_i = -b - K w_i moves the result as  R_total Exp(sum_i P_i^T delta_i dt_i), P_i = S_{i+1}..S_n
 => r_k(theta) = r0_k + J_k theta,  theta=[b(3), vec(K)(9)].   All J_k computed once; any window = a small linear LS."""
import sys
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from common import *
OUT = "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle/"

def interval_terms(t, gs, t0, t1):
    """returns (Rg (3,3), Jb (3,3), JK (3,9), wbar (3,)) for one interval."""
    m = (t > t0) & (t < t1)
    ts = np.concatenate(([t0], t[m], [t1])).astype(np.int64)
    w = np.stack([np.interp(ts, t, gs[:, i]) for i in range(3)], 1)
    dt = np.diff(ts) / 1e9
    wm = 0.5 * (w[:-1] + w[1:])
    S = Rotation.from_rotvec(wm * dt[:, None]).as_matrix()          # (n,3,3)
    n = len(S)
    # P_i = S_{i+1}...S_n  (P_{n-1} = I)
    P = np.empty_like(S); acc = np.eye(3)
    for i in range(n - 1, -1, -1):
        P[i] = acc; acc = S[i] @ acc
    Rg = acc                                                          # S_0 ... S_{n-1}
    PT = np.swapaxes(P, 1, 2)                                         # (n,3,3)
    Jb = -np.einsum("nab,n->ab", PT, dt)
    # JK[:, 3a + c] = -sum_i PT[i][:, a] * wm[i, c] * dt[i]
    JK = -np.einsum("nra,nc,n->rac", PT, wm, dt).reshape(3, 9)
    return Rg, Jb, JK, (wm * dt[:, None]).sum(0) / (ts[-1] - ts[0]) * 1e9

def build(name, ctrl, tau=0.5, save=True):
    rdir = rec_dir(name)
    t, gb, ab = load_imu(rdir, ctrl)
    gs = (DIAG_FLIP @ gb.T).T
    mo = MocapOrientation(load_mocap_device(rdir, ctrl))
    dtn = int(tau * 1e9)
    grid = np.arange(t[0] + 0.5e9, t[-1] - 0.5e9, dtn).astype(np.int64)
    ok = mo.valid(grid) & mo.valid(grid + dtn) & mo.valid(grid + dtn // 2)
    grid = grid[ok]
    R0 = mo.R_world_imu(grid); R1 = mo.R_world_imu(grid + dtn)
    T0, R, JB, JKs, WB = [], [], [], [], []
    for i, t0 in enumerate(grid):
        Rg, Jb, JK, wbar = interval_terms(t, gs, int(t0), int(t0) + dtn)
        Rm = (R0[i].inv() * R1[i]).as_matrix()
        R.append(Rotation.from_matrix(Rm.T @ Rg).as_rotvec()); JB.append(Jb); JKs.append(JK); T0.append(t0); WB.append(wbar)
    d = dict(t0=np.array(T0), r0=np.array(R), Jb=np.array(JB), JK=np.array(JKs), wbar=np.array(WB), tau=tau)
    if save: np.savez(OUT + f"gyro_terms_{name}_{ctrl}_tau{tau}.npz", **d)
    return d

def solve(d, idx, with_K=True, K_fixed=None, sigma=None):
    """LS over intervals idx.  returns theta (b, K) ; if K_fixed given solve only b."""
    r0 = d["r0"][idx]; Jb = d["Jb"][idx]; JK = d["JK"][idx]
    if K_fixed is not None:
        y = -(r0 + JK @ K_fixed.reshape(9)) ; A = Jb
        keep = np.ones(len(idx), bool)
        for _ in range(3):
            b, *_ = np.linalg.lstsq(A[keep].reshape(-1, 3), y[keep].reshape(-1), rcond=None)
            res = (y - A @ b); s = 1.4826 * np.median(np.abs(res[keep])) + 1e-12
            keep = (np.abs(res) < 4 * s).all(1)
        return b, K_fixed, keep
    A = np.concatenate([Jb, JK], axis=2) if with_K else Jb
    y = -r0
    keep = np.ones(len(idx), bool)
    for _ in range(3):
        th, *_ = np.linalg.lstsq(A[keep].reshape(-1, A.shape[2]), y[keep].reshape(-1), rcond=None)
        res = y - A @ th; s = 1.4826 * np.median(np.abs(res[keep])) + 1e-12
        keep = (np.abs(res) < 4 * s).all(1)
    return (th[:3], (th[3:].reshape(3, 3) if with_K else np.zeros((3, 3))), keep)

if __name__ == "__main__":
    tau = 0.5
    for name in (sys.argv[1:] or ["static_dark"]):
        for ctrl in CTRLS:
            d = build(name, ctrl, tau)
            idx = np.arange(len(d["t0"]))
            b0, _, k0 = solve(d, idx, with_K=False)
            b1, K1, k1 = solve(d, idx, with_K=True)
            res0 = np.linalg.norm(d["r0"], axis=1)
            print(f"{name}/{ctrl}: n={len(idx)} tau={tau}s  |wbar| median {np.median(np.linalg.norm(d['wbar'],axis=1)):.2f} rad/s")
            print(f"   bias-only fit   b = {b0.round(4)} rad/s (kept {k0.sum()})")
            print(f"   bias+K fit      b = {b1.round(4)} rad/s  K diag {np.diag(K1).round(4)} offdiag max {np.abs(K1-np.diag(np.diag(K1))).max():.4f} (kept {k1.sum()})")
            print(f"   median interval residual before {np.median(res0):.4f} rad ({np.degrees(np.median(res0)):.2f} deg); after bias+K: {np.degrees(np.median(np.linalg.norm(d['r0'][idx]+np.concatenate([d['Jb'],d['JK']],2)[idx]@np.concatenate([b1,K1.reshape(9)]),axis=1))):.2f} deg")
