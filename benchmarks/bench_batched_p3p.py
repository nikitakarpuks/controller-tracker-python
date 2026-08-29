"""
Batched closed-form P3P (Grunert's quartic formulation) vs. N individual
cv2.solveP3P calls, to isolate how much of solveP3P's per-call cost is fixed
Python<->C++ dispatch overhead vs. genuine per-problem math.

The batched solver processes ALL N triples with exactly two native calls
total, regardless of N:
  - one np.linalg.eigvals call on a stack of N companion matrices (solves
    all N quartics for u at once)
  - one np.linalg.svd call on a stack of N*4 covariance matrices (solves
    the absolute-orientation / Kabsch step for every (triple, candidate-root)
    pair at once)
vs. cv2.solveP3P's N separate Python<->C++ round trips (one per triple).

Correctness is validated first (rotation/translation compared directly
against cv2.solveP3P's own output, not just internally self-consistent) --
a batched-and-wrong implementation would be a meaningless speed number.

Run: python benchmarks/bench_batched_p3p.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.bench_p3p_blob import load_camera, load_led_positions

RNG = np.random.default_rng(0)


def bearing_from_norm(pts2):
    """(...,2) normalised (identity-K) image coords -> (...,3) unit bearing vectors."""
    ones = np.ones(pts2.shape[:-1] + (1,), dtype=pts2.dtype)
    v = np.concatenate([pts2, ones], axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def solve_p3p_batch(world_pts, img_norm):
    """
    world_pts : (N,3,3) three world-frame points per triple
    img_norm  : (N,3,2) three normalised (identity-K) image points per triple

    Returns (R, t, valid): R (N,4,3,3), t (N,4,3), valid (N,4) bool -- up to
    4 candidate solutions per triple, mirroring cv2.solveP3P's own "up to 4
    geometric solutions" contract (no 4th-point disambiguation here, same as
    cv2.solveP3P itself -- that happens downstream in the real pipeline).
    """
    N = world_pts.shape[0]
    P1, P2, P3 = world_pts[:, 0], world_pts[:, 1], world_pts[:, 2]
    f = bearing_from_norm(img_norm)
    f1, f2, f3 = f[:, 0], f[:, 1], f[:, 2]

    a = np.linalg.norm(P2 - P3, axis=-1)
    b = np.linalg.norm(P1 - P3, axis=-1)
    c = np.linalg.norm(P1 - P2, axis=-1)

    cos_alpha = np.sum(f2 * f3, axis=-1)
    cos_beta = np.sum(f1 * f3, axis=-1)
    cos_gamma = np.sum(f1 * f2, axis=-1)

    p = (b / c) ** 2
    q = (a / c) ** 2

    m2 = q - p - 1
    m1 = 2 * cos_gamma * (p - q)
    m0 = 1 - p + q
    d1c = -2 * cos_alpha
    d0c = 2 * cos_beta
    B1 = -2 * cos_beta
    c2 = -p
    c1 = 2 * p * cos_gamma
    c0 = 1 - p

    A4 = m2 ** 2 + c2 * d1c ** 2
    A3 = 2 * m2 * m1 + B1 * m2 * d1c + 2 * c2 * d1c * d0c + c1 * d1c ** 2
    A2 = (2 * m2 * m0 + m1 ** 2) + B1 * (m2 * d0c + m1 * d1c) + c2 * d0c ** 2 + 2 * c1 * d1c * d0c + c0 * d1c ** 2
    A1 = 2 * m1 * m0 + B1 * (m1 * d0c + m0 * d1c) + c1 * d0c ** 2 + 2 * c0 * d1c * d0c
    A0 = m0 ** 2 + B1 * m0 * d0c + c0 * d0c ** 2

    # One batched eigvals call solves all N quartics (companion-matrix roots).
    A4s = np.where(np.abs(A4) < 1e-12, 1e-12, A4)
    comp = np.zeros((N, 4, 4), dtype=np.float64)
    comp[:, 0, :] = np.stack([-A3 / A4s, -A2 / A4s, -A1 / A4s, -A0 / A4s], axis=-1)
    comp[:, 1, 0] = 1.0
    comp[:, 2, 1] = 1.0
    comp[:, 3, 2] = 1.0
    roots = np.linalg.eigvals(comp)  # (N,4) complex

    real_mask = np.abs(roots.imag) < 1e-6
    u_all = roots.real  # (N,4)

    Nu = m2[:, None] * u_all ** 2 + m1[:, None] * u_all + m0[:, None]
    Du = d1c[:, None] * u_all + d0c[:, None]
    Du_safe = np.where(np.abs(Du) < 1e-9, np.nan, Du)
    v_all = Nu / Du_safe

    denom = 1 + u_all ** 2 - 2 * u_all * cos_gamma[:, None]
    d1_sq = (c[:, None] ** 2) / np.where(denom <= 1e-9, np.nan, denom)
    with np.errstate(invalid="ignore"):
        d1_all = np.sqrt(np.where(d1_sq > 0, d1_sq, np.nan))
    d2_all = u_all * d1_all
    d3_all = v_all * d1_all

    valid = (real_mask & np.isfinite(d1_all) & np.isfinite(v_all)
             & (d1_all > 0) & (d2_all > 0) & (d3_all > 0))

    # Candidate camera-frame 3-point sets for every (triple, root): (N,4,3,3)
    Pc = np.stack([
        d1_all[..., None] * f1[:, None, :],
        d2_all[..., None] * f2[:, None, :],
        d3_all[..., None] * f3[:, None, :],
    ], axis=2)
    Pw = np.broadcast_to(world_pts[:, None, :, :], (N, 4, 3, 3))

    # One batched svd call solves absolute orientation for all N*4 candidates.
    Pw_f = Pw.reshape(N * 4, 3, 3)
    Pc_f = np.nan_to_num(Pc.reshape(N * 4, 3, 3))
    Pw_mean = Pw_f.mean(axis=1, keepdims=True)
    Pc_mean = Pc_f.mean(axis=1, keepdims=True)
    Xc = Pw_f - Pw_mean
    Yc = Pc_f - Pc_mean
    H = np.einsum("mij,mik->mjk", Xc, Yc)
    U, S, Vt = np.linalg.svd(H)
    V = np.swapaxes(Vt, -1, -2)
    Ut = np.swapaxes(U, -1, -2)
    R_raw = V @ Ut
    d = np.sign(np.linalg.det(R_raw))
    V2 = V.copy()
    V2[:, :, -1] *= d[:, None]
    R = V2 @ Ut
    t = Pc_mean[:, 0, :] - np.einsum("mij,mj->mi", R, Pw_mean[:, 0, :])

    return R.reshape(N, 4, 3, 3), t.reshape(N, 4, 3), valid


# ---------------------------------------------------------------------------
# Correctness validation against cv2.solveP3P, per-triple
# ---------------------------------------------------------------------------

def make_triple(K, dc, led_positions):
    idx = RNG.choice(len(led_positions), size=3, replace=False)
    world = led_positions[idx].astype(np.float64)
    rvec = RNG.uniform(-0.3, 0.3, size=3)
    tvec = np.array([RNG.uniform(-0.1, 0.1), RNG.uniform(-0.1, 0.1), RNG.uniform(0.3, 1.0)])
    img_dist, _ = cv2.fisheye.projectPoints(world.reshape(3, 1, 3), rvec, tvec, K, dc)
    img_norm = cv2.fisheye.undistortPoints(img_dist, K, dc).reshape(3, 2)
    return world, img_norm


def validate(n_triples=200):
    K, dc = load_camera()
    led_positions = load_led_positions()
    worlds = np.empty((n_triples, 3, 3))
    imgs = np.empty((n_triples, 3, 2))
    for i in range(n_triples):
        worlds[i], imgs[i] = make_triple(K, dc, led_positions)

    R_batch, t_batch, valid = solve_p3p_batch(worlds, imgs)

    n_sols_mismatch = 0
    max_rot_err_deg = 0.0
    max_trans_err_m = 0.0
    matched = 0
    total_cv_sols = 0

    for i in range(n_triples):
        n_sols, rvecs, tvecs = cv2.solveP3P(
            worlds[i].reshape(3, 1, 3).astype(np.float32),
            imgs[i].reshape(3, 1, 2).astype(np.float32),
            np.eye(3, dtype=np.float32), np.zeros(4, dtype=np.float32),
            flags=cv2.SOLVEPNP_P3P,
        )
        n_valid_batch = int(valid[i].sum())
        if n_valid_batch != n_sols:
            n_sols_mismatch += 1

        for rv, tv in zip(rvecs, tvecs):
            total_cv_sols += 1
            Rc, _ = cv2.Rodrigues(rv)
            best_rot_err, best_trans_err = None, None
            for k in range(4):
                if not valid[i, k]:
                    continue
                cos_ang = np.clip((np.trace(Rc.T @ R_batch[i, k]) - 1) / 2, -1, 1)
                rot_err = np.degrees(np.arccos(cos_ang))
                trans_err = np.linalg.norm(tv.flatten() - t_batch[i, k])
                if best_rot_err is None or rot_err < best_rot_err:
                    best_rot_err, best_trans_err = rot_err, trans_err
            if best_rot_err is not None and best_rot_err < 0.5 and best_trans_err < 0.01:
                matched += 1
                max_rot_err_deg = max(max_rot_err_deg, best_rot_err)
                max_trans_err_m = max(max_trans_err_m, best_trans_err)

    print("=== Correctness validation vs cv2.solveP3P ===")
    print(f"  triples tested            : {n_triples}")
    print(f"  solution-count mismatches : {n_sols_mismatch} / {n_triples}")
    print(f"  cv2 solutions matched     : {matched} / {total_cv_sols}")
    print(f"  max rotation error (deg)  : {max_rot_err_deg:.5f}")
    print(f"  max translation error (m) : {max_trans_err_m:.6f}")


# ---------------------------------------------------------------------------
# Timing: N individual cv2.solveP3P calls vs. one solve_p3p_batch call
# ---------------------------------------------------------------------------

def bench_timing():
    K, dc = load_camera()
    led_positions = load_led_positions()
    K_id_f32, dc0_f32 = np.eye(3, dtype=np.float32), np.zeros(4, dtype=np.float32)

    print("\n=== N individual cv2.solveP3P calls vs. 1 batched numpy call ===")
    print(f"{'N':>6}  {'N x cv2 (ms)':>14}  {'batched (ms)':>14}  {'speedup':>9}")
    for n in (50, 100, 500, 1000, 2000):
        worlds = np.empty((n, 3, 3), dtype=np.float64)
        imgs = np.empty((n, 3, 2), dtype=np.float64)
        for i in range(n):
            worlds[i], imgs[i] = make_triple(K, dc, led_positions)
        worlds_f32 = worlds.astype(np.float32)
        imgs_f32 = imgs.astype(np.float32)

        def loop_cv2():
            for i in range(n):
                cv2.solveP3P(
                    worlds_f32[i].reshape(3, 1, 3), imgs_f32[i].reshape(3, 1, 2),
                    K_id_f32, dc0_f32, flags=cv2.SOLVEPNP_P3P,
                )

        def batched():
            solve_p3p_batch(worlds, imgs)

        for _ in range(2):
            loop_cv2(); batched()  # warmup

        reps = 20 if n <= 500 else 5
        t0 = time.perf_counter()
        for _ in range(reps):
            loop_cv2()
        ms_cv2 = (time.perf_counter() - t0) / reps * 1000

        t0 = time.perf_counter()
        for _ in range(reps):
            batched()
        ms_batch = (time.perf_counter() - t0) / reps * 1000

        print(f"{n:>6}  {ms_cv2:>14.2f}  {ms_batch:>14.2f}  {ms_cv2/ms_batch:>8.1f}x")


if __name__ == "__main__":
    validate(n_triples=200)
    bench_timing()
