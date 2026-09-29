"""fastgap.py -- O(1)-per-method evaluation of gyro rotation-prediction error for a constant bias over a gap.
For gap [ta,tb] integrate the gyro ONCE with b=0 storing dR0 = dR_g(0) and C = sum_s dt_s R_{s->end}^T so that
    dR_g(b) ~= dR0 . Exp(-C b)      (first order in b*dt; commutator error ~ |omega||b|dt^2 per sample, negligible)
and the prediction error rotation is  E(b) = dR_g(b)^T dR_v = Exp(C b) . (dR0^T dR_v).   Verified against exact integration in test_fastgap()."""
import numpy as np
from scipy.spatial.transform import Rotation
from common import integrate_gyro_segment, rot_deg


def integrate_with_C(t_gyro, gyro, ts0, ts1):
    """Same discretisation as src.imu_data.integrate_gyro_segment (linear interp of gyro, midpoint rule) + the C matrix."""
    if ts1 <= ts0 or ts0 < t_gyro[0] or ts1 > t_gyro[-1]:
        return None, None
    mask = (t_gyro > ts0) & (t_gyro < ts1)
    ts = np.concatenate(([ts0], t_gyro[mask], [ts1])).astype(np.int64)
    om = np.stack([np.interp(ts, t_gyro, gyro[:, k]) for k in range(3)], 1)
    dts = np.diff(ts) / 1e9
    w = 0.5 * (om[:-1] + om[1:])
    steps = Rotation.from_rotvec(w * dts[:, None]).as_matrix()          # A_s
    n = len(steps)
    # suffix products S_s = A_s ... A_n  (rotation from start of step s to end)
    suffix = np.empty_like(steps)
    acc = np.eye(3)
    for s in range(n - 1, -1, -1):
        acc = steps[s] @ acc
        suffix[s] = acc
    R_total = suffix[0]
    # R_{(s+1)->end} = suffix[s+1] (identity for the last step)
    after = np.concatenate([suffix[1:], np.eye(3)[None]], axis=0)
    C = np.einsum("s,sji->ij", dts, after)                               # sum dt_s (R_{(s+1)->end})^T
    return R_total, C


def fast_error_deg(dR0, C, dR_v, b):
    E0 = dR0.T @ dR_v
    return rot_deg(Rotation.from_rotvec(C @ b).as_matrix() @ E0)


def test_fastgap(t_gyro, gyro, ts_list, dRv_list, rng=np.random.default_rng(1)):
    worst = 0.0
    for (ta, tb), dRv in zip(ts_list, dRv_list):
        dR0, C = integrate_with_C(t_gyro, gyro, ta, tb)
        if dR0 is None: continue
        b = rng.uniform(-0.03, 0.03, 3)
        exact = rot_deg(integrate_gyro_segment(t_gyro, gyro - b, ta, tb).T @ dRv)
        fast = fast_error_deg(dR0, C, dRv, b)
        worst = max(worst, abs(exact - fast))
    return worst
