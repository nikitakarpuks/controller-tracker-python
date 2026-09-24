"""Accelerometer bias oracle from mocap (position level, no differentiation of noisy mocap).
Per window: p_m(t) = p0 + v0 tau + dd[ R f_meas + g0 ] + sum_j theta_j U_j(t),  model f_true = f_meas - b - S f_meas, g = g0 + dg.
   U_bi = -dd[R e_i],   U_S[a,c] = -dd[R e_a f_c],   U_dg_i = tau^2/2 e_i  ; optional lever arm r: +dd[R (alpha x e_i + w x (w x e_i))].
 p0,v0 are nuisance per window (projected out).  Mocap IMU frame == sensor frame; mocap 'IMU' position assumed = accelerometer position (tested via lever-arm columns)."""
import sys
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from common import *
from scipy.signal import savgol_filter
OUT = "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle/"
G0 = np.array([0.0, -9.81, 0.0])   # src.imu_data.MOCAP_ROOM_G_WORLD

class MocapPose(MocapOrientation):
    def pos_imu(self, q_ns):
        tl = np.clip(self.lookup_times(q_ns), self.t_mocap[0], self.t_mocap[-1])
        pm = np.stack([np.interp(tl, self.t_mocap, self.dev.position[:, i]) for i in range(3)], 1)
        Rm = self.slerp(tl)
        return pm + Rm.apply(-(self.R_im.T @ self.dev.T_imu_marker.t))

def cum2(t_s, x):
    """double trapezoid integral of x (n,k) over t_s (n,) starting at 0; returns (n,k)."""
    dt = np.diff(t_s)[:, None]
    v = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(0.5 * (x[1:] + x[:-1]) * dt, 0)])
    return np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(0.5 * (v[1:] + v[:-1]) * dt, 0)])

def window_system(t, fs, w_s, mo, dev_t_ns, T0, W, use_lever=False):
    """returns (A (3N,P), y (3N,), Nuis (3N,6)) or None.  P = 3 (b) + 9 (S) + 3 (dg) [+3 r]."""
    sel = np.flatnonzero((t >= T0) & (t <= T0 + W))
    Wsec = W / 1e9
    if len(sel) < Wsec * 150: return None
    tI = t[sel].astype(np.float64); f = fs[sel]; w = w_s[sel]
    if not (mo.valid(np.array([T0, T0 + W // 2, T0 + W])).all()): return None
    RI = mo.R_world_imu(tI.astype(np.int64)).as_matrix()                         # (n,3,3) R_world_sensor
    ts = (tI - tI[0]) / 1e9
    # mocap sample times inside window (mocap-track time -> camera time via local constant shift)
    shift = mo.lookup_times(np.array([T0 + W // 2]))[0] - (T0 + W // 2)
    mt = dev_t_ns[(dev_t_ns - shift >= tI[0]) & (dev_t_ns - shift <= tI[-1])]
    if len(mt) < Wsec * 100: return None
    q = (mt - shift).astype(np.int64)
    if np.diff(mt).max() > 30e6: return None
    pm = mo.pos_imu(q); tau_m = (q - tI[0]) / 1e9
    a0 = np.einsum("nij,nj->ni", RI, f) + G0
    cols = []
    cols.append(a0)                                                              # base
    for i in range(3): cols.append(-RI[:, :, i])                                 # bias
    for a in range(3):
        for c in range(3): cols.append(-RI[:, :, a] * f[:, c:c+1])               # S[a,c]
    for i in range(3):
        e = np.zeros((len(tI), 3)); e[:, i] = 1.0; cols.append(e)                # dg
    if use_lever:
        wd = savgol_filter(w, 7, 2, deriv=1, delta=np.median(np.diff(ts)), axis=0)
        for i in range(3):
            ei = np.zeros(3); ei[i] = 1.0
            lin = np.cross(wd, ei) + np.cross(w, np.cross(w, ei))
            cols.append(np.einsum("nij,nj->ni", RI, lin))
    U = [cum2(ts, c) for c in cols]
    U = [np.stack([np.interp(tau_m, ts, u[:, k]) for k in range(3)], 1) for u in U]   # (N,3) each
    y = (pm - U[0]).reshape(-1)
    A = np.stack([u.reshape(-1) for u in U[1:]], 1)
    N = len(tau_m); Nu = np.zeros((3 * N, 6))
    for k in range(3):
        Nu[k::3, k] = 1.0; Nu[k::3, 3 + k] = tau_m
    return A, y, Nu

def project(A, y, Nu):
    Qn, _ = np.linalg.qr(Nu)
    return A - Qn @ (Qn.T @ A), y - Qn @ (Qn.T @ y)

def run(name, ctrl, W_s=2.0, use_lever=False, cache=True):
    rdir = rec_dir(name); dev = load_mocap_device(rdir, ctrl); mo = MocapPose(dev)
    t, gb, ab = load_imu(rdir, ctrl); fs = (DIAG_FLIP @ ab.T).T; w_s = (DIAG_FLIP @ gb.T).T
    W = int(W_s * 1e9); wins = []
    for T0 in np.arange(t[0] + 0.5e9, t[-1] - W - 0.5e9, W):
        r = window_system(t, fs, w_s, mo, dev.t_ns, int(T0), W, use_lever)
        if r is None: continue
        A, y, Nu = project(*r[:2], r[2]) if False else (None, None, None)
        Ap, yp = project(r[0], r[1], r[2])
        wins.append(dict(t0=int(T0), AtA=Ap.T @ Ap, Aty=Ap.T @ yp, yty=float(yp @ yp), n=len(yp), Am=np.abs(r[1]).mean()))
    return wins

def solve_from(wins, idx, P_sel=None, fixed=None, ridge=0.0):
    AtA = sum(wins[i]["AtA"] for i in idx); Aty = sum(wins[i]["Aty"] for i in idx); yty = sum(wins[i]["yty"] for i in idx); n = sum(wins[i]["n"] for i in idx)
    P = AtA.shape[0]
    th = np.linalg.solve(AtA + ridge * np.eye(P), Aty)
    rss = yty - 2 * th @ Aty + th @ AtA @ th
    cov = (rss / max(n - P, 1)) * np.linalg.inv(AtA + ridge * np.eye(P))
    return th, np.sqrt(np.diag(cov)), np.sqrt(rss / n)

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
    for ctrl in CTRLS:
        for lever in (False, True):
            wins = run(name, ctrl, 2.0, lever)
            idx = list(range(len(wins)))
            th, se, rms = solve_from(wins, idx)
            # zero-parameter residual for reference
            rms0 = np.sqrt(sum(w["yty"] for w in wins) / sum(w["n"] for w in wins))
            print(f"{name}/{ctrl} lever={lever}: {len(wins)} windows; rms pos misfit {rms0*1000:.2f} mm (no model) -> {rms*1000:.2f} mm (fit)")
            print(f"    b_a = {th[0:3].round(4)} +/- {se[0:3].round(4)} m/s^2   dg = {th[12:15].round(4)}   S diag {np.array([th[3],th[7],th[11]]).round(4)}  offdiag max {np.abs(th[3:12].reshape(3,3)-np.diag(np.diag(th[3:12].reshape(3,3)))).max():.4f}" + (f"   lever r = {(th[15:18]*1000).round(1)} mm" if lever else ""))
