import math
import time

import cv2
import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment, least_squares
from scipy.spatial.distance import cdist
from itertools import combinations, product
from typing import Dict, List, Optional, Tuple

from scipy.spatial import KDTree

from src.geometry import _compute_geometry
from src.debug_config import is_deep, get_debug_triple, is_verbose_all, log_best
from src._pnp import _ransac_pnp, _project_points, _check_z_range
from src._visibility import _visible_mask, _cross_occluded_mask
from src.transformations import Transform


# ── LED graph helpers (from _led_graph.py) ──────────────────

def _build_led_neighbor_lists(positions: np.ndarray, normals: np.ndarray, k: int = 8) -> List[np.ndarray]:
    """
    For each LED (anchor): among LEDs whose normal is within 90° of the anchor's normal
    (dot product >= 0), return up to k nearest by Euclidean distance.

    Normal filter first — only LEDs facing roughly the same direction are candidates,
    ensuring they can be simultaneously visible.  Spatial sort second — closest
    normal-compatible LEDs form the tightest, most discriminative hypotheses.
    """
    n = len(positions)
    k_act = min(k, n - 1)
    dists = cdist(positions, positions)   # (N, N)
    dots  = normals @ normals.T           # (N, N) pairwise normal cosines

    result = []
    for i in range(n):
        valid = dots[i] >= 0.0
        valid[i] = False
        candidates = np.where(valid)[0]
        if len(candidates) == 0:
            result.append(np.array([], dtype=int))
            continue
        order = np.argsort(dists[i, candidates])
        result.append(candidates[order[:k_act]])
    return result


def _build_led_neighbor_lists_edge(
    positions: np.ndarray,
    normals: np.ndarray,
    is_inner: np.ndarray,
    z_rel: np.ndarray,
    k: int = 8,
) -> List[np.ndarray]:
    """
    Alternative neighbourhood for grazing-angle views (~30° to the frustum base plane)
    where both inner and outer LEDs are simultaneously visible.

    For each anchor LED the k neighbours are filled with a strict split:
      - n_same  = min(k // 2, 2): at most 2 nearest same-type LEDs (outer→outer or
                                   inner→inner) with dot >= 0, sorted by distance.
      - n_cross = k - n_same     : cross-type LEDs (outer→inner or inner→outer)
                                   with dot >= 0, sorted by normal similarity descending.

    The two halves are interleaved with cross-type first:
      cross[0], same[0], cross[1], same[1], …
    so rank 0 is always the best cross-type match (the grazing-view target), rank 1 the
    nearest same-type, and deeper ranks continue alternating.  Depth-2 triple (0,1)
    therefore pairs the best cross LED with the nearest same LED.

    debug_led_ids: if provided, print the chosen neighbours for each listed LED id.
    """
    n       = len(positions)
    n_same  = min(k // 2, 2)
    n_cross = k - n_same

    dists = cdist(positions, positions)   # (N,N) Euclidean distances
    dots  = normals @ normals.T           # (N,N) pairwise normal cosines

    result = []
    for i in range(n):
        # ── same-type: nearest by distance, dot >= 0, capped at 2 ────────────────
        same_valid = (is_inner == is_inner[i]) & (dots[i] >= 0.0)
        same_valid[i] = False
        same_cands = np.where(same_valid)[0]
        if len(same_cands):
            same_nbrs = same_cands[np.argsort(dists[i, same_cands])[:n_same]]
        else:
            same_nbrs = np.array([], dtype=int)

        # ── cross-type: most normal-similar, dot >= 0 ────────────────────────────
        cross_valid = (is_inner != is_inner[i]) & (dots[i] >= 0.0)
        cross_cands = np.where(cross_valid)[0]
        if len(cross_cands):
            cross_nbrs = cross_cands[np.argsort(-dots[i, cross_cands])[:n_cross]]
        else:
            cross_nbrs = np.array([], dtype=int)

        # ── interleave: cross[0], same[0], cross[1], same[1], … ─────────────────
        nbrs = []
        for slot in range(max(len(same_nbrs), len(cross_nbrs))):
            if slot < len(cross_nbrs):
                nbrs.append(cross_nbrs[slot])
            if slot < len(same_nbrs):
                nbrs.append(same_nbrs[slot])
        nbrs = np.array(nbrs, dtype=int)
        result.append(nbrs)

    return result


def _build_blob_neighbor_lists(blobs: np.ndarray, k: int) -> List[np.ndarray]:
    """
    For each blob: indices of up to k nearest blob neighbours, excluding self.
    """
    n = len(blobs)
    if n <= 1:
        return [np.array([], dtype=int) for _ in range(n)]
    tree  = KDTree(blobs)
    k_act = min(k, n - 1)
    _, idx = tree.query(blobs, k=k_act + 1)
    return [row[1:] for row in idx]


def _precompute_led_quads(
    positions: np.ndarray, led_nbr: List[np.ndarray], k: int = 8
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Enumerate LED triples for P3P. Each (anchor, l1, l2) ordered triple appears once per
    anchor — i.e., the same three LEDs may appear under different anchors because each
    anchor defines a distinct P3P geometry (different gate pool, different depth rank).
    Duplicate bijection-level P3P calls are deduplicated later in brute_match.

    Returns
    -------
    triple_idx   : (N, 3) int32      — [anchor, l1, l2]; all three used for P3P
    triple_depth : (N,) int32        — max neighbour rank used (1-based); shallow/deep split
    triple_gates : List[np.ndarray]  — for each triple, remaining neighbour LED indices (gate pool)
    """
    idx_rows:   List[Tuple]       = []
    depth_rows: List[int]         = []
    gate_rows:  List[np.ndarray]  = []
    seen_per_anchor: set          = set()   # dedup only within the same anchor's C(k,2)
    n = len(positions)
    for anchor in range(n):
        nbrs   = led_nbr[anchor][:k]
        nb_len = len(nbrs)
        if nb_len < 2:
            continue
        for i1, i2 in combinations(range(nb_len), 2):
            l1, l2 = int(nbrs[i1]), int(nbrs[i2])
            key = (anchor, min(l1, l2), max(l1, l2))
            if key in seen_per_anchor:
                continue
            seen_per_anchor.add(key)
            depth  = i2 + 1
            gates  = np.array(
                [int(nbrs[j]) for j in range(nb_len) if j != i1 and j != i2],
                dtype=np.int32,
            )
            idx_rows.append((anchor, l1, l2))
            depth_rows.append(depth)
            gate_rows.append(gates)
    if not idx_rows:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            [],
        )
    return (
        np.array(idx_rows,   dtype=np.int32),
        np.array(depth_rows, dtype=np.int32),
        gate_rows,
    )


# ── Brute-search gate helpers (from _matching.py module-level) ──────────

def _gate_any_point(
    R_h: np.ndarray, tvec_h: np.ndarray,
    gate_obj: np.ndarray,
    gate_img: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    thresh_sq: float,
    track_dist: bool = False,
) -> Tuple[bool, float]:
    """
    Return True if ANY gate LED projects within sqrt(thresh_sq) pixels of ANY gate blob.
    If either pool is empty, returns True (no gate to fail).
    Returns (passed, min_dist_px); min_dist is only tracked when track_dist=True.
    """
    if len(gate_obj) == 0 or len(gate_img) == 0:
        return True, 0.0

    min_dist = np.inf if track_dist else 0.0
    for obj in gate_obj:
        p = R_h @ obj + tvec_h
        if p[2] <= 0:
            continue
        iz = 1.0 / p[2]
        px = fx * p[0] * iz + cx
        py = fy * p[1] * iz + cy
        for img in gate_img:
            dx = px - img[0]
            dy = py - img[1]
            if track_dist:
                min_dist = min(min_dist, math.sqrt(dx * dx + dy * dy))
            if dx * dx + dy * dy <= thresh_sq:
                return True, min_dist
    return False, min_dist


def _gate_fourth_point(
    R_h: np.ndarray, tvec_h: np.ndarray,
    obj4: np.ndarray, img4: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    p4_thresh_sq: float,
) -> bool:
    """
    Fast single-point reprojection gate (no distortion, inline perspective divide).
    Return True if the 4th LED projects within sqrt(p4_thresh_sq) pixels of img4.
    """
    p4_cam = R_h @ obj4 + tvec_h
    if p4_cam[2] <= 0:
        return False
    iz = 1.0 / p4_cam[2]
    dx = fx * p4_cam[0] * iz + cx - img4[0]
    dy = fy * p4_cam[1] * iz + cy - img4[1]
    return dx * dx + dy * dy <= p4_thresh_sq


def _tier_label(t):
    nbr = t[2] if len(t) > 2 else 'standard'
    return f"led≤{t[0]}, blob≤{t[1]}, nbr={nbr}"


# ---------------------------------------------------------------------------
# blob LED-ID carry helper
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Joint multi-camera LM refinement
# ---------------------------------------------------------------------------

def _filter_aux_by_reprojection(
    T_world_ctrl,
    aux_joint_data: List,   # [(Camera, blobs_np, [(blob_idx, led_id)])]
    positions: np.ndarray,
    threshold_px: float,
) -> List:
    """Return aux_joint_data with each camera's pairs pre-filtered to reprojection < threshold_px."""
    result = []
    for _ocam, _oblobs, _pairs in aux_joint_data:
        if not _pairs:
            continue
        _T_ci = _ocam.T_world_cam.inverse().compose(T_world_ctrl)
        _rv_ci, _ = cv2.Rodrigues(_T_ci.R.astype(np.float32))
        _tv_ci = _T_ci.t.astype(np.float32)
        _led_ids   = [led_id for _, led_id in _pairs]
        _blob_idxs = [b_idx  for b_idx, _  in _pairs]
        _pts3d = positions[_led_ids].astype(np.float32)
        _proj  = _project_points(_rv_ci, _tv_ci, _pts3d, _ocam.camera_matrix, _ocam.dist_coeffs,
                                 is_fisheye=_ocam.is_fisheye)
        _errs  = np.linalg.norm(_proj - _oblobs[_blob_idxs], axis=1)
        _kept  = [_pairs[i] for i in range(len(_pairs)) if _errs[i] < threshold_px]
        if _kept:
            result.append((_ocam, _oblobs, _kept))
    return result


def _joint_refine_pose(
    T_world_ctrl_init: Transform,
    primary_cam,
    primary_pairs: List[Tuple[int, int]],   # [(blob_idx, led_id)]
    primary_blobs: np.ndarray,
    aux_cam_data: List,                      # [(Camera, blobs_np, [(blob_j, led_id)])]
    model_positions: np.ndarray,
    max_nfev: int = 200,
    primary_weight: float = 2.0,
    huber_scale: float = 1.5,
) -> Tuple[Optional[Transform], float]:
    """
    Refine T_world_ctrl jointly over all cameras via Trust-Region with Huber loss.

    Parameterisation: 6-vector [rvec(3) | tvec(3)] for T_world_ctrl.
    Residuals: (proj_x - blob_x, proj_y - blob_y) per correspondence per camera,
    scaled by a per-camera weight so cameras contribute equally regardless of
    correspondence count. Primary camera gets an additional `primary_weight`
    multiplier to keep the RANSAC-validated fit dominant.

    Huber loss (f_scale=huber_scale) down-weights aux outliers without
    discarding them entirely — mismatched aux pairs influence the solution
    much less than valid ones.

    Returns (refined_T_world_ctrl, unweighted_mean_reproj_err_px) or (None, inf).
    """
    # Pre-assemble immutable per-camera arrays once — avoids object creation in the hot loop.
    cam_data = []
    for cam, blobs_c, pairs_c in (
        [(primary_cam, primary_blobs, primary_pairs)] +
        [(ac, ab, ap) for ac, ab, ap in aux_cam_data if ap]
    ):
        if not pairs_c:
            continue
        T_cw = cam.T_world_cam.inverse()
        b_idx = np.array([b for b, _ in pairs_c], dtype=np.int32)
        l_idx = np.array([l for _, l in pairs_c], dtype=np.int32)
        cam_data.append((
            T_cw.R.astype(np.float64),          # R_cam_world
            T_cw.t.astype(np.float64),          # t_cam_world
            cam.camera_matrix,
            cam.dist_coeffs,
            model_positions[l_idx].astype(np.float32),
            blobs_c[b_idx].astype(np.float32),
            getattr(cam, "is_fisheye", False),
        ))

    if sum(len(e[4]) for e in cam_data) < 4:
        return None, np.inf

    # Per-camera residual weight: normalise by 1/sqrt(n_pairs) so each camera
    # contributes equally to the total objective regardless of correspondence count.
    # Primary gets an additional `primary_weight` multiplier.
    cam_weights = np.array([
        (primary_weight if i == 0 else 1.0) / float(np.sqrt(max(len(e[4]), 1)))
        for i, e in enumerate(cam_data)
    ], dtype=np.float64)

    # Flat weight vector aligned with the residual vector — used to recover
    # unweighted errors for reporting after optimisation.
    flat_weights = np.concatenate([
        np.full(2 * len(e[4]), w) for e, w in zip(cam_data, cam_weights)
    ])

    rvec0, _ = cv2.Rodrigues(T_world_ctrl_init.R.astype(np.float32))
    x0 = np.concatenate([rvec0.ravel(), T_world_ctrl_init.t]).astype(np.float64)

    def _residuals(x):
        rv = x[:3].astype(np.float32).reshape(3, 1)
        tv = x[3:].astype(np.float64)
        R_wc = cv2.Rodrigues(rv)[0].astype(np.float64)
        parts = []
        for (R_cw, t_cw, K_c, dc_c, leds, blobs_ref, is_fe), w in zip(cam_data, cam_weights):
            R_ci = (R_cw @ R_wc).astype(np.float32)
            t_ci = (R_cw @ tv + t_cw).astype(np.float32)
            rv_ci = cv2.Rodrigues(R_ci)[0]
            if is_fe:
                proj, _ = cv2.fisheye.projectPoints(
                    leds.astype(np.float64).reshape(-1, 1, 3),
                    rv_ci.astype(np.float64), t_ci.astype(np.float64).reshape(3, 1),
                    K_c.astype(np.float64), dc_c,
                )
            else:
                proj, _ = cv2.projectPoints(
                    leds.reshape(-1, 1, 3), rv_ci, t_ci.reshape(3, 1), K_c, dc_c,
                )
            parts.append(((proj.reshape(-1, 2) - blobs_ref) * w).ravel())
        return np.concatenate(parts).astype(np.float64)

    try:
        result = least_squares(
            _residuals, x0, method='trf',
            loss='huber', f_scale=huber_scale,
            ftol=1e-7, xtol=1e-7, gtol=1e-7, max_nfev=max_nfev,
        )
    except Exception:
        return None, np.inf

    tv_opt = result.x[3:].astype(np.float64)
    if np.linalg.norm(tv_opt) > 10.0:     # sanity: reject absurd translations
        return None, np.inf

    R_opt = cv2.Rodrigues(result.x[:3].astype(np.float32).reshape(3, 1))[0].astype(np.float64)
    T_opt = Transform(R_opt, tv_opt)
    # Divide weighted residuals by flat_weights to recover unweighted pixel errors.
    mean_err = float(np.mean(np.linalg.norm(
        (result.fun / flat_weights).reshape(-1, 2), axis=1
    )))
    return T_opt, mean_err


class PoseSearcher:
    """
    Stateless (read-only after __init__) pose estimator for one (camera, model) pair.

    Holds all geometry and precomputed LED data that the three search strategies
    need.  The strategies are attached as real class methods via Python's descriptor
    protocol — no __get__ hack required.

    Thread-safe: instances carry no mutable state after construction, so the same
    PoseSearcher can be called from multiple threads (different frames) once
    parallelisation is added in Step 5.
    """

    def __init__(self, camera, model, geometry_cfg: dict = None, matching_cfg: dict = None):
        self.camera        = camera
        self.model         = model
        self.T_world_cam   = camera.T_world_cam
        self._matching_cfg = matching_cfg or {}

        self._cam  = camera.camera_idx
        self._ctrl = model.name.replace("_controller", "")

        _cfg = matching_cfg or {}
        # ── cached config (static for the lifetime of this searcher) ────────────
        # shared
        self._c_facing_deg          = float(_cfg.get('led_facing_angle_deg',             86.0))
        self._c_led_radius_mm       = float(_cfg.get('self._c_led_radius_mm',                     2.5))
        self._c_blob_size_min       = float(_cfg.get('self._c_blob_size_min',              0.2))
        self._c_blob_size_max       = float(_cfg.get('self._c_blob_size_max',              4.0))
        self._c_blob_size_score_w   = float(_cfg.get('self._c_blob_size_score_w',            0.5))
        self._c_blob_bright_score_w = float(_cfg.get('self._c_blob_bright_score_w',      0.3))
        self._c_occ_radius          = float(_cfg.get('cross_occlusion_bounding_radius_m', 0.18))
        self._c_occ_margin_px       = float(_cfg.get('cross_occlusionself._c_occ_margin_px',   20.0))
        self._c_log_size_filter     = bool( _cfg.get('log_size_filter',                  False))
        # joint optimisation
        self._c_joint_opt           = bool( _cfg.get('joint_optimization',               True))
        self._c_joint_prefilter_px  = float(_cfg.get('joint_aux_prefilter_px',            8.0))
        self._c_joint_primary_w     = float(_cfg.get('joint_primary_weight',              2.0))
        self._c_joint_huber_scale   = float(_cfg.get('joint_huber_scale',                1.5))
        # proximity
        self._c_prox_reproj_px      = float(_cfg.get('proximity_reprojection_threshold',  2.0))
        self._c_prox_min_inliers    = int(  _cfg.get('proximity_min_inliers',
                                                      _cfg.get('min_inliers',            4)))
        self._c_prox_expansion_px   = float(_cfg.get('proximity_expansion_px',            8.0))
        self._c_prox_max_hyp        = int(  _cfg.get('proximity_max_hypotheses',          256))
        self._c_prox_none_penalty    = float(_cfg.get('proximity_none_penalty_px',        0.3))
        self._c_prox_none_factor     = float(_cfg.get('proximity_none_penalty_factor',    1.0))
        self._c_prox_strong_match_px = float(_cfg.get('proximity_strong_match_px',        0.2))
        self._c_prox_branch_k       = int(  _cfg.get('proximity_branch_k',                3))
        self._c_prox_level0_max_hyp = int(  _cfg.get('proximity_level0_max_hyp',          16))
        self._c_prox_none_branch_w  = int(  _cfg.get('proximity_none_branch_width',        5))
        self._c_prox_score_metric   = str(  _cfg.get('proximity_score_metric',           'max'))
        self._c_prox_top_k_ransac        = int(  _cfg.get('proximity_top_k_ransac',           3))
        self._c_vis_occlusion_margin_m   = float(_cfg.get('visibility_occlusion_margin_m',  0.0))
        self._c_prox_vis_score_threshold = float(_cfg.get('proximity_vis_score_threshold',  0.95))
        # constrained
        self._c_cs_reproj_px        = float(_cfg.get('reprojection_threshold',            2.0))
        self._c_cs_snap_factor      = float(_cfg.get('proximity_self._c_cs_snap_factor',             4.0))
        # brute-force
        _brute_reproj               = float(_cfg.get('reprojection_threshold',            1.5))
        self._c_brute_depth_tiers   = tuple(tuple(t) for t in _cfg.get('depth_tiers',
                                        ((2, 3), (2, 4), (2, 4, 'edge'),
                                         (3, 5), (3, 5, 'edge'), (4, 6))))
        self._c_brute_p4_px         = float(_cfg.get('p4_threshold_px',                  2.0))
        self._c_brute_hungarian_px  = float(_cfg.get('hungarian_threshold_px',            5.0))
        self._c_brute_reproj_px     = _brute_reproj
        self._c_brute_min_inliers   = int(  _cfg.get('min_inliers',                      4))
        self._c_brute_min_frac      = _cfg.get('min_inlier_fraction',               None) or None
        self._c_brute_strong_in     = int(  _cfg.get('strong_match_inliers',             7))
        self._c_brute_strong_err    = float(_cfg.get('strong_match_error_px',            1.5))
        self._c_brute_min_vis_cov   = float(_cfg.get('min_vis_coverage',                 0.75))
        self._c_brute_rng_seed      = _cfg.get('rng_seed',                              42)
        self._c_brute_aux_reproj_px = float(_cfg.get('joint_aux_prefilter_px',
                                                      _brute_reproj * 2.0))

        positions = model.positions.astype("float32")
        normals   = model.normals.astype("float32")

        self._geometry = _compute_geometry(positions, normals, geometry_cfg)

        self._led_nbr = _build_led_neighbor_lists(positions, normals)
        self._led_triple_idx, self._led_triple_depth, self._led_triple_gates = (
            _precompute_led_quads(positions, self._led_nbr)
        )
        self._led_nbr_edge = _build_led_neighbor_lists_edge(
            positions, normals, self._geometry.is_inner, self._geometry.z_rel
        )
        self._led_triple_idx_edge, self._led_triple_depth_edge, self._led_triple_gates_edge = (
            _precompute_led_quads(positions, self._led_nbr_edge)
        )

    @staticmethod
    def _dbg(active: bool, msg: str) -> None:
        if active:
            logger.debug(msg)

    def _aux_cam_vis(
        self,
        ocam,
        T_world_ctrl,
        occluders_per_cam: Optional[Dict],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        """Return (R, t, rv, focal_px, vis_mask) for an aux camera given a world-frame controller pose."""
        _T_ci = ocam.T_world_cam.inverse().compose(T_world_ctrl)
        R     = _T_ci.R.astype(np.float32)
        t     = _T_ci.t.astype(np.float32)
        rv    = cv2.Rodrigues(R)[0]
        focal = float(max(ocam.camera_matrix[0, 0], ocam.camera_matrix[1, 1]))
        vis_scores = _visible_mask(
            R, t, self.model.positions, self.model.normals, self._geometry,
            cam_K=ocam.camera_matrix, cam_dc=ocam.dist_coeffs,
            cam_w=ocam.width, cam_h=ocam.height, cam_rpmax=ocam.rpmax,
            cam_is_fisheye=ocam.is_fisheye,
            facing_threshold_deg=self._c_facing_deg,
        )
        if occluders_per_cam:
            occ = occluders_per_cam.get(ocam.camera_idx)
            if occ is not None:
                R_occ, t_occ, geom_occ = occ
                _cross_occ = _cross_occluded_mask(
                    R, t, self.model.positions,
                    R_occ, t_occ, geom_occ,
                    self._c_occ_radius, self._c_occ_radius, focal, self._c_occ_margin_px,
                    log_tag=f"[{self._ctrl} | cam {self._cam} aux_cam {ocam.camera_idx}]",
                    vis_mask=vis_scores > 0.0,
                )
                vis_scores[_cross_occ] = 0.0
        return R, t, rv, focal, vis_scores >= 1.0

    def _snap_led_pairs(
        self,
        lids: List[int],
        proj: np.ndarray,
        led_cam_pts: np.ndarray,
        blobs: np.ndarray,
        blob_radii: Optional[np.ndarray],
        focal_px: float,
    ) -> List[Tuple[int, int]]:
        """
        Snap each LED (by id in lids) to the nearest eligible blob.
        Returns (blob_idx, led_id) pairs with no blob used twice.
        """
        pairs: List[Tuple[int, int]] = []
        used: set = set()
        for ii, lid in enumerate(lids):
            depth       = float(max(led_cam_pts[ii, 2], 0.01))
            expected_px = focal_px * (self._c_led_radius_mm / 1000.0) / depth
            snap_px     = expected_px * self._c_cs_snap_factor
            dists       = np.linalg.norm(blobs - proj[ii], axis=1)
            if blob_radii is not None:
                elig = ((blob_radii >= expected_px * self._c_blob_size_min) &
                        (blob_radii <= expected_px * self._c_blob_size_max))
                if elig.any():
                    cand   = np.where(elig)[0]
                    scores = dists[cand] + self._c_blob_size_score_w * np.abs(blob_radii[cand] - expected_px)
                    j      = int(cand[np.argmin(scores)])
                else:
                    j = int(np.argmin(dists))
            else:
                j = int(np.argmin(dists))
            if dists[j] < snap_px and j not in used:
                pairs.append((j, lid))
                used.add(j)
        return pairs

    def _apply_joint_lm(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        primary_pairs: List[Tuple[int, int]],
        blobs: np.ndarray,
        aux_assignments: Dict[int, List],
        other_cameras_blobs: List,
        K: np.ndarray,
        dc: np.ndarray,
        label: str,
        prior_error: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Joint multi-camera LM refinement over all cameras simultaneously.
        Returns (rvec_refined, tvec_refined, error_refined) or None if skipped/failed.
        """
        if not self._c_joint_opt or not aux_assignments or not other_cameras_blobs:
            return None
        _aux = [
            (ocam, np.asarray(oblobs, dtype=np.float32), aux_assignments.get(ocam.camera_idx, []))
            for ocam, oblobs, _ in other_cameras_blobs
            if aux_assignments.get(ocam.camera_idx)
        ]
        if not _aux:
            return None
        _R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32).reshape(3, 1))
        _T_wc = self.T_world_cam.compose(
            Transform(_R.astype(np.float64), np.asarray(tvec, dtype=np.float64).reshape(3))
        )
        _aux = _filter_aux_by_reprojection(_T_wc, _aux, self.model.positions, self._c_joint_prefilter_px)
        if not _aux:
            return None
        _T_joint, _joint_err = _joint_refine_pose(
            _T_wc, self.camera, primary_pairs, blobs, _aux, self.model.positions,
            primary_weight=self._c_joint_primary_w,
            huber_scale=self._c_joint_huber_scale,
        )
        if _T_joint is None:
            return None
        _T_prim = self.T_world_cam.inverse().compose(_T_joint)
        _rv_j   = cv2.Rodrigues(_T_prim.R.astype(np.float32))[0]
        _tv_j   = _T_prim.t.astype(np.float32)
        _lo     = self.model.positions[[l for _, l in primary_pairs]].astype(np.float32)
        _li     = blobs[[b for b, _ in primary_pairs]].astype(np.float32)
        _err_j  = float(np.mean(np.linalg.norm(_project_points(_rv_j, _tv_j, _lo, K, dc,
                                                                is_fisheye=self.camera.is_fisheye) - _li, axis=1)))
        logger.debug(
            f"[{self._ctrl} | cam {self._cam}] {label} joint LM: "
            f"primary err {prior_error:.2f}→{_err_j:.2f}px  joint mean {_joint_err:.2f}px"
        )
        for _aux_cam_j, _aux_blobs_j, _aux_pr_j in _aux:
            if not _aux_pr_j:
                continue
            _T_aux_j  = _aux_cam_j.T_world_cam.inverse().compose(_T_joint)
            _rv_aux_j = cv2.Rodrigues(_T_aux_j.R.astype(np.float32))[0]
            _tv_aux_j = _T_aux_j.t.astype(np.float32)
            _leds_j   = self.model.positions[[l for _, l in _aux_pr_j]].astype(np.float32)
            _blobs_j  = _aux_blobs_j[[b for b, _ in _aux_pr_j]].astype(np.float32)
            _err_ax   = float(np.mean(np.linalg.norm(
                _project_points(_rv_aux_j, _tv_aux_j, _leds_j,
                                _aux_cam_j.camera_matrix, _aux_cam_j.dist_coeffs,
                                is_fisheye=_aux_cam_j.is_fisheye) - _blobs_j,
                axis=1,
            )))
            logger.debug(
                f"[{self._ctrl} | cam {self._cam}] {label} joint LM cam{_aux_cam_j.camera_idx}: "
                f"err={_err_ax:.2f}px  ({len(_aux_pr_j)} pairs)"
            )
        return _rv_j, _tv_j, _err_j

    def proximity_search(
        self,
        blobs: np.ndarray,
        predicted_pose: Tuple[np.ndarray, np.ndarray],
        blob_brightnesses: Optional[np.ndarray] = None,
        other_cameras_blobs: Optional[List] = None,
        occluders_per_cam: Optional[Dict[int, Tuple[np.ndarray, np.ndarray, object]]] = None,
        expansion_px: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Refine a predicted pose via Hungarian matching over model-visible LEDs.

        Projects visible LEDs with the predicted pose, identifies locked pairs (exactly
        one blob candidate per LED), refines the pose with a quick PnP on those pairs,
        then runs Hungarian on the refined projections. RANSAC filters inliers.
        Returns None when too few pairs survive; caller falls back to brute_match.
        """
        rvec_pred, tvec_pred = predicted_pose
        rvec_pred = np.asarray(rvec_pred, dtype=np.float32).reshape(3, 1)
        tvec_pred = np.asarray(tvec_pred, dtype=np.float32).reshape(3)

        _t_collisions = _t_scoring = _t_stage2 = _t_aux = _t_lm = 0.0

        K  = self.camera.camera_matrix
        dc = self.camera.dist_coeffs

        geom = self._geometry
        proximity_expansion_px = expansion_px if expansion_px is not None else self._c_prox_expansion_px

        R_pred_arr, _ = cv2.Rodrigues(rvec_pred)
        focal_px = float(max(K[0, 0], K[1, 1]))

        vis_scores_pred = _visible_mask(
            R_pred_arr, tvec_pred,
            self.model.positions, self.model.normals,
            geom,
            cam_K=K, cam_dc=dc, cam_w=self.camera.width, cam_h=self.camera.height,
            cam_rpmax=self.camera.rpmax, cam_is_fisheye=self.camera.is_fisheye,
            facing_threshold_deg=self._c_facing_deg,
            occlusion_margin_m=self._c_vis_occlusion_margin_m,
        )
        if occluders_per_cam:
            _occ = occluders_per_cam.get(self.camera.camera_idx)
            if _occ is not None:
                _R_occ, _t_occ, _geom_occ = _occ
                _cross_occ = _cross_occluded_mask(
                    R_pred_arr, tvec_pred, self.model.positions,
                    _R_occ, _t_occ, _geom_occ,
                    self._c_occ_radius, self._c_occ_radius, focal_px, self._c_occ_margin_px,
                    log_tag=f"[{self._ctrl} | cam {self._cam}]",
                    vis_mask=vis_scores_pred > 0.0,
                )
                vis_scores_pred[_cross_occ] = 0.0
        vis_mask_pred   = vis_scores_pred >= self._c_prox_vis_score_threshold
        n_model_visible = int(vis_mask_pred.sum())

        _brightness_norm = None
        if blob_brightnesses is not None and len(blob_brightnesses) > 0:
            _bmax = float(blob_brightnesses.max())
            if _bmax > 0:
                _brightness_norm = blob_brightnesses / _bmax

        # Hypothesis testing: per-LED candidate sets, enumerate valid assignments,
        # score each by solvePnP reprojection error, pick the best.
        pairs      = []
        locked_obj = []
        locked_img = []
        _locked_pose_dbg = None   # DEBUG: (rvec, tvec) from PnP on truly-locked pairs only

        if n_model_visible > 0 and len(blobs) > 0:
            vis_ids  = np.where(vis_mask_pred)[0]
            vis_pos  = self.model.positions[vis_ids].astype(np.float32)
            proj_vis = _project_points(rvec_pred, tvec_pred, vis_pos, K, dc,
                                       is_fisheye=self.camera.is_fisheye)

            # Step 1: per-LED candidate blobs within expansion_px, sorted by distance.
            dist_pred  = cdist(proj_vis, blobs)
            candidates = []
            for k in range(len(vis_ids)):
                in_neigh = np.where(dist_pred[k] < proximity_expansion_px)[0]
                if len(in_neigh) > 0:
                    candidates.append(in_neigh[np.argsort(dist_pred[k, in_neigh])].tolist())
                else:
                    candidates.append([])

            # Step 2: compute how many LED neighbourhoods each blob appears in.
            blob_led_count: Dict[int, int] = {}
            for cands in candidates:
                for b in cands:
                    blob_led_count[b] = blob_led_count.get(b, 0) + 1

            # Step 3: partition.
            # truly_locked_k : 1 candidate AND that blob is unique to this LED's neighbourhood.
            # hyp_k          : 2+ candidates OR 1 candidate shared with another LED's neighbourhood.
            # Empty LEDs (0 candidates) are skipped entirely.
            truly_locked_k: List[int] = []
            hyp_k:          List[int] = []
            for k, cands in enumerate(candidates):
                if len(cands) == 0:
                    continue
                elif len(cands) == 1:
                    if blob_led_count[int(cands[0])] == 1:
                        truly_locked_k.append(k)
                    else:
                        hyp_k.append(k)   # shared blob — must compete via hypothesis
                else:
                    hyp_k.append(k)

            truly_locked_blobs = set(int(candidates[k][0]) for k in truly_locked_k)
            n_empty   = sum(1 for c in candidates if len(c) == 0)
            n_shared1 = sum(1 for k in hyp_k if len(candidates[k]) == 1)
            n_ambig   = len(hyp_k) - n_shared1

            locked_led_ids = [int(vis_ids[k]) for k in truly_locked_k]
            logger.debug(
                f"[{self._ctrl} | cam {self._cam}] Proximity candidates: "
                f"locked={len(truly_locked_k)}  shared1={n_shared1}  ambiguous={n_ambig}  empty={n_empty}"
                f"  expansion={proximity_expansion_px:.1f}px"
                f"  locked_ids={locked_led_ids}"
                + (f"  hyp_sizes={[len(candidates[k]) for k in hyp_k]}" if hyp_k else "")
            )

            # Pre-undistort all blob positions once — reused across all hypothesis evaluations.
            _blobs_inp = blobs.astype(np.float64).reshape(-1, 1, 2)
            if self.camera.is_fisheye:
                _blobs_norm = cv2.fisheye.undistortPoints(_blobs_inp, K.astype(np.float64), dc).reshape(-1, 2)
            else:
                _blobs_norm = cv2.undistortPoints(_blobs_inp.astype(np.float32), K, dc).reshape(-1, 2)

            # DEBUG: PnP on truly-locked pairs only, for pose-comparison logging below.
            # Not used for any downstream processing.
            if len(truly_locked_k) >= 3:
                _lk_obj  = vis_pos[truly_locked_k]
                _lk_bidx = np.array([int(candidates[k][0]) for k in truly_locked_k], dtype=np.int32)
                _lk_norm = _blobs_norm[_lk_bidx]
                try:
                    _lk_ok, _lk_rv, _lk_tv = cv2.solvePnP(
                        _lk_obj.astype(np.float64).reshape(-1, 1, 3),
                        _lk_norm.astype(np.float64).reshape(-1, 1, 2),
                        np.eye(3, dtype=np.float64), np.zeros(4, dtype=np.float64),
                        rvec_pred.astype(np.float64).copy(),
                        tvec_pred.astype(np.float64).reshape(3, 1).copy(),
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                except Exception:
                    _lk_ok = False
                if _lk_ok:
                    _locked_pose_dbg = (np.asarray(_lk_rv).ravel(), np.asarray(_lk_tv).ravel())

            def _score_hyp(hyp_blob_indices: list) -> Tuple[float, Dict[int, float]]:
                """solvePnP on truly-locked + matched hypothesis pairs; returns
                (aggregate reprojection error per proximity_score_metric,
                {hyp_position: residual_px}). The residual map lets callers rank
                which non-None hyp position fits worst, without a second solve.
                None entries mean 'no blob for this LED' and are skipped. Falls
                back to pixel distance from predicted projection if PnP fails."""
                _agg = np.max if self._c_prox_score_metric == 'max' else np.mean
                matched_pos = [i for i in range(len(hyp_k)) if hyp_blob_indices[i] is not None]
                matched_hyp = [(hyp_k[i], int(hyp_blob_indices[i])) for i in matched_pos]
                all_led_k   = truly_locked_k + [k for k, _ in matched_hyp]
                all_blob_c  = [int(candidates[k][0]) for k in truly_locked_k] + [b for _, b in matched_hyp]
                n_locked    = len(truly_locked_k)

                if not all_led_k:
                    return float('inf'), {}

                all_obj = vis_pos[all_led_k]
                all_img = blobs[np.array(all_blob_c, dtype=np.int32)]

                if len(all_led_k) >= 4:
                    pts_norm = _blobs_norm[np.array(all_blob_c, dtype=np.int32)]
                    try:
                        ok_h, rvec_h, tvec_h = cv2.solvePnP(
                            all_obj.astype(np.float64).reshape(-1, 1, 3),
                            pts_norm.astype(np.float64).reshape(-1, 1, 2),
                            np.eye(3, dtype=np.float64), np.zeros(4, dtype=np.float64),
                            rvec_pred.astype(np.float64).copy(),
                            tvec_pred.astype(np.float64).reshape(3, 1).copy(),
                            useExtrinsicGuess=True,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                        )
                        if ok_h:
                            proj_h = _project_points(
                                np.asarray(rvec_h, dtype=np.float32).reshape(3, 1),
                                np.asarray(tvec_h, dtype=np.float32).reshape(3),
                                all_obj, K, dc, is_fisheye=self.camera.is_fisheye,
                            )
                            resid = np.linalg.norm(proj_h - all_img, axis=1)
                            resid_map = {p: float(resid[n_locked + j]) for j, p in enumerate(matched_pos)}
                            return float(_agg(resid)), resid_map
                    except Exception:
                        pass

                # PnP unavailable or failed: pixel distance from predicted projection.
                resid = np.linalg.norm(proj_vis[all_led_k] - all_img, axis=1)
                resid_map = {p: float(resid[n_locked + j]) for j, p in enumerate(matched_pos)}
                return float(_agg(resid)), resid_map

            def _fmt_leds(blobs_list):
                return "[" + ", ".join(
                    f"{int(vis_ids[hyp_k[i]])}:{'✓' if blobs_list[i] is not None else 'None'}"
                    for i in range(len(hyp_k))
                ) + "]"

            # Step 4: enumerate hypotheses and pick the best.
            # None is appended to each hyp LED's candidate list as a valid "no match" option.
            if not hyp_k:
                best_hyp_blobs = []
                logger.debug(
                    f"[{self._ctrl} | cam {self._cam}] Proximity: no hypothesis LEDs, "
                    f"direct assignment of {len(truly_locked_k)} locked pairs"
                )
            else:
                # Score: error + none_penalty_px * n_none — a match is preferred only if
                # it reduces reprojection error by more than the per-None penalty.
                best_key       = float('inf')
                best_none      = 0
                best_hyp_blobs = None
                n_tried        = 0
                n_collisions   = 0
                # Top-K candidates for stage-2 RANSAC rescoring (sorted ascending by key).
                top_candidates: list = []

                def _bt_combos(n_none_target: int):
                    """Backtracking generator: collision-free combos with exactly n_none_target Nones.
                    Prunes duplicate-blob branches immediately — no post-filter needed."""
                    _n    = len(hyp_k)
                    _used = set(truly_locked_blobs)
                    _co   = [None] * _n

                    def _bt(i: int, nones_left: int):
                        if i == _n:
                            yield list(_co)
                            return
                        remaining = _n - i
                        # Assign a blob — skip if all remaining slots must be None
                        if nones_left < remaining:
                            for _b in candidates[hyp_k[i]]:
                                if _b not in _used:
                                    _co[i] = _b; _used.add(_b)
                                    yield from _bt(i + 1, nones_left)
                                    _used.discard(_b); _co[i] = None
                        # Assign None — skip if no Nones left to spend
                        if nones_left > 0:
                            _co[i] = None
                            yield from _bt(i + 1, nones_left - 1)

                    yield from _bt(0, n_none_target)

                # Minimum None count forced by blob scarcity: if fewer distinct blobs
                # are available than hyp LEDs, a full (0-None) assignment is impossible.
                # Skip all levels below this minimum to avoid iterating only collisions.
                _avail_blobs = (
                    set(b for k in hyp_k for b in candidates[k]) - truly_locked_blobs
                )
                _min_none_forced = max(0, len(hyp_k) - len(_avail_blobs))

                # Pre-collect 0-None combos and sort by total raw pixel distance so the
                # best-by-individual-distance combo is tried first, enabling earlier
                # best_key establishment and faster None-cost pruning.
                _zero_none_combos: list = []
                if _min_none_forced == 0:
                    _t0 = time.perf_counter()
                    for _vals in product(*[candidates[hyp_k[i]] for i in range(len(hyp_k))]):
                        _combo = list(_vals)
                        _ms = set(_combo)
                        if len(_ms) < len(_combo) or _ms & truly_locked_blobs:
                            n_collisions += 1
                            continue
                        _raw = sum(dist_pred[hyp_k[i], _combo[i]] for i in range(len(hyp_k)))
                        _zero_none_combos.append((_raw, _combo))
                    _t_collisions += time.perf_counter() - _t0
                    _zero_none_combos.sort(key=lambda x: x[0])
                    _zero_none_total = len(_zero_none_combos)
                    if self._c_prox_level0_max_hyp > 0 and _zero_none_total > self._c_prox_level0_max_hyp:
                        logger.debug(
                            f"[{self._ctrl} | cam {self._cam}] Proximity: level-0 cap "
                            f"kept {self._c_prox_level0_max_hyp}/{_zero_none_total} combos by raw distance"
                        )
                        _zero_none_combos = _zero_none_combos[:self._c_prox_level0_max_hyp]
                _had_zero_none_candidates = len(_zero_none_combos) > 0
                # Nones at the base level are unavoidable and carry no penalty;
                # only extras above it are penalised.
                n_none_base = (
                    0 if _had_zero_none_candidates
                    else (_min_none_forced if _min_none_forced > 0 else None)
                )

                _stopped_early = False
                _first_valid_level_seen = _had_zero_none_candidates
                _level_parents: list = []  # (key, combo, resid_map) — top-K from the previous None level

                def _log_hyp(n_tried, combo_blobs, score, key, best_key):
                    if is_deep():
                        _err_label = f"{self._c_prox_score_metric}_err"
                        _leds_fmt = "[" + ", ".join(
                            f"{int(vis_ids[hyp_k[i]])}:{'✓' if combo_blobs[i] is not None else 'None'}"
                            for i in range(len(hyp_k))
                        ) + "]"
                        logger.debug(
                            f"[{self._ctrl} | cam {self._cam}] Proximity hyp {n_tried}: "
                            f"leds={_leds_fmt}  {_err_label}={score:.2f}px"
                            + ("  ← best" if key < best_key else "")
                        )

                def _update_top_k(key, n_none, combo_blobs):
                    _tk = self._c_prox_top_k_ransac
                    if _tk >= 2:
                        if len(top_candidates) < _tk:
                            top_candidates.append((key, n_none, list(combo_blobs)))
                            top_candidates.sort(key=lambda x: x[0])
                        elif key < top_candidates[-1][0]:
                            top_candidates[-1] = (key, n_none, list(combo_blobs))
                            top_candidates.sort(key=lambda x: x[0])

                # --- Level 0: process pre-sorted 0-None combos ---
                for _raw, combo_blobs in _zero_none_combos:
                    _t0 = time.perf_counter(); score, resid_map = _score_hyp(combo_blobs); _t_scoring += time.perf_counter() - _t0
                    key   = score  # none_cost == 0
                    _log_hyp(n_tried, combo_blobs, score, key, best_key)
                    if key < best_key:
                        best_key = key; best_none = 0; best_hyp_blobs = combo_blobs
                    _update_top_k(key, 0, combo_blobs)
                    _level_parents.append((key, list(combo_blobs), resid_map))
                    n_tried += 1

                    if n_tried >= self._c_prox_max_hyp:
                        logger.debug(
                            f"[{self._ctrl} | cam {self._cam}] Proximity: hypothesis cap "
                            f"({self._c_prox_max_hyp}) reached  ({n_collisions} collisions skipped)"
                        )
                        break

                    if score <= self._c_prox_strong_match_px:
                        _stopped_early = True
                        logger.debug(
                            f"[{self._ctrl} | cam {self._cam}] Proximity: early stop "
                            f"(0-None score {score:.2f}px <= {self._c_prox_strong_match_px:.2f}px)"
                        )
                        break

                _level_parents.sort(key=lambda x: x[0])
                _level_parents = _level_parents[:self._c_prox_branch_k]

                # --- Levels 1+ : branch from top-K parents of the previous level ---
                if not _stopped_early and n_tried < self._c_prox_max_hyp:
                    for n_none in range(max(1, _min_none_forced), len(hyp_k) + 1):
                        if n_none_base is not None:
                            _n_extra = n_none - n_none_base
                            none_cost = self._c_prox_none_penalty * sum(
                                self._c_prox_none_factor ** _k for _k in range(_n_extra)
                            )
                            if none_cost >= best_key:
                                break
                        else:
                            none_cost = 0.0  # base not yet known; no pruning

                        _used_fallback = not bool(_level_parents)
                        if _level_parents:
                            # Each parent spawns children only for its weakest-fit matched
                            # positions (by the parent's own per-point PnP residual), not
                            # every matched position — narrows branching to the LED(s)
                            # actually likely to be a bad assignment.
                            _seen_ck: set = set()
                            _this_level: list = []
                            _bw = self._c_prox_none_branch_w
                            for _, _par, _resid in _level_parents:
                                _matched = [_i for _i in range(len(_par)) if _par[_i] is not None]
                                _matched.sort(key=lambda _i: _resid.get(_i, 0.0), reverse=True)
                                for _i in _matched[:_bw]:
                                    _child = list(_par); _child[_i] = None
                                    _ck = tuple(_child)
                                    if _ck not in _seen_ck:
                                        _seen_ck.add(_ck); _this_level.append(_child)
                        else:
                            # Fallback: no parents — backtracking enumerates only collision-free
                            # combos, so no post-filter is needed.
                            _this_level = []
                            _t0 = time.perf_counter()
                            for _combo in _bt_combos(n_none):
                                _this_level.append(_combo)
                            _t_collisions += time.perf_counter() - _t0

                        if not _this_level:
                            if _used_fallback:
                                continue  # this n_none has no valid combos; try next level
                            break         # branching: empty children → no grandchildren either

                        _is_first_available = not _first_valid_level_seen
                        _first_valid_level_seen = True
                        if n_none_base is None:
                            n_none_base = n_none

                        _next_parents: list = []
                        _cap_hit = False
                        for combo_blobs in _this_level:
                            _t0 = time.perf_counter(); score, resid_map = _score_hyp(combo_blobs); _t_scoring += time.perf_counter() - _t0
                            key   = score + none_cost
                            _log_hyp(n_tried, combo_blobs, score, key, best_key)
                            if key < best_key:
                                best_key = key; best_none = n_none; best_hyp_blobs = combo_blobs
                            _update_top_k(key, n_none, combo_blobs)
                            _next_parents.append((key, list(combo_blobs), resid_map))
                            n_tried += 1

                            if n_tried >= self._c_prox_max_hyp:
                                logger.debug(
                                    f"[{self._ctrl} | cam {self._cam}] Proximity: hypothesis cap "
                                    f"({self._c_prox_max_hyp}) reached  ({n_collisions} collisions skipped)"
                                )
                                _cap_hit = True
                                break

                            if _is_first_available and score <= self._c_prox_strong_match_px:
                                _stopped_early = True
                                logger.debug(
                                    f"[{self._ctrl} | cam {self._cam}] Proximity: early stop "
                                    f"({n_none}-None score {score:.2f}px <= {self._c_prox_strong_match_px:.2f}px)"
                                )
                                _cap_hit = True
                                break

                        _next_parents.sort(key=lambda x: x[0])
                        _level_parents = _next_parents[:self._c_prox_branch_k]
                        if _cap_hit:
                            break

                if best_hyp_blobs is None:
                    best_hyp_blobs = []
                    logger.debug(
                        f"[{self._ctrl} | cam {self._cam}] Proximity: all {n_tried + n_collisions} combos invalid"
                        f"  ({n_collisions} collisions), using truly-locked pairs only"
                    )
                else:
                    best_none_cost = self._c_prox_none_penalty * sum(
                        self._c_prox_none_factor ** k for k in range(best_none - n_none_base)
                    )
                    best_err  = best_key - best_none_cost
                    n_matched = len(hyp_k) - best_none
                    _err_label = f"best_{self._c_prox_score_metric}_err"
                    logger.debug(
                        f"[{self._ctrl} | cam {self._cam}] Proximity: {n_tried} hypotheses evaluated"
                        f"  {n_collisions} skipped (collision)"
                        f"  {_err_label}={best_err:.2f}px  matched={n_matched}/{len(hyp_k)} hyp LEDs"
                        f"  best_hyp_leds={_fmt_leds(best_hyp_blobs)}"
                    )

                    # Stage-2: RANSAC rescore the top-K hypotheses.
                    # ITERATIVE solvePnP inflates the error when one pair is a noisy
                    # blob that is still a valid inlier; RANSAC recovers the true
                    # quality by ignoring that pair during scoring.  We run RANSAC only
                    # on the K cheapest ITERATIVE hypotheses (default K=3) — not all 64.
                    #
                    # Scoring mirrors the ITERATIVE key:
                    #   ransac_key = mean_inlier_error
                    #              + none_penalty * Σ none_factor^k  (k=0..n_extra-1)
                    # where n_extra = (hyp_Nones + RANSAC_drops) - n_none_base.
                    # n_none_base is the minimum None count forced by blob scarcity;
                    # those Nones are unavoidable and carry no penalty.  Only Nones
                    # above the base — whether from the hypothesis or RANSAC drops —
                    # are penalised, using the same geometric series as the main loop.
                    if self._c_prox_top_k_ransac >= 2 and len(top_candidates) >= 2:
                        _t_stage2_start = time.perf_counter()
                        _s2_best_key   = float('inf')
                        _s2_best_blobs = best_hyp_blobs
                        _s2_best_none  = best_none

                        for _rk, (_rkey, _rn_none, _rblobs) in enumerate(top_candidates):
                            _locked_r = [(int(candidates[k][0]), int(vis_ids[k]))
                                         for k in truly_locked_k]
                            _hyp_r    = [(int(_rblobs[i]), int(vis_ids[hyp_k[i]]))
                                         for i in range(len(_rblobs))
                                         if _rblobs[i] is not None]
                            _pairs_r  = _locked_r + _hyp_r
                            if len(_pairs_r) < self._c_prox_min_inliers:
                                logger.debug(
                                    f"[{self._ctrl} | cam {self._cam}] Proximity stage-2 rank-{_rk+1}: "
                                    f"iter_key={_rkey:.2f}  leds={_fmt_leds(_rblobs)}  SKIP (too few pairs)"
                                )
                                continue
                            _o_r = self.model.positions[[l for _, l in _pairs_r]].astype(np.float32)
                            _i_r = blobs[[b for b, _ in _pairs_r]].astype(np.float32)
                            _ok_r, _rv_r, _tv_r, _idx_r = _ransac_pnp(
                                _o_r, _i_r, K, dc, rvec_pred, tvec_pred,
                                reprojection_px=self._c_prox_reproj_px,
                                is_fisheye=self.camera.is_fisheye,
                            )
                            if not _ok_r or _idx_r is None:
                                logger.debug(
                                    f"[{self._ctrl} | cam {self._cam}] Proximity stage-2 rank-{_rk+1}: "
                                    f"iter_key={_rkey:.2f}  leds={_fmt_leds(_rblobs)}  RANSAC failed"
                                )
                                continue
                            _n_r        = len(_idx_r)
                            _n_drop_r   = len(_pairs_r) - _n_r
                            _n_missing_r = _rn_none + _n_drop_r   # hyp Nones + RANSAC drops
                            _proj_r     = _project_points(_rv_r, _tv_r, _o_r[_idx_r], K, dc,
                                                          is_fisheye=self.camera.is_fisheye)
                            _err_r      = float(np.linalg.norm(_proj_r - _i_r[_idx_r], axis=1).mean())
                            _miss_cost_r = self._c_prox_none_penalty * sum(
                                self._c_prox_none_factor ** _dk for _dk in range(_n_missing_r - n_none_base)
                            )
                            _s2_key_r = _err_r + _miss_cost_r
                            _marker   = "  <- best" if (_s2_key_r, _rn_none) < (_s2_best_key, _s2_best_none) else ""
                            logger.debug(
                                f"[{self._ctrl} | cam {self._cam}] Proximity stage-2 rank-{_rk+1}: "
                                f"iter_key={_rkey:.2f}  leds={_fmt_leds(_rblobs)}  "
                                f"inliers={_n_r}/{len(_pairs_r)}  "
                                f"err={_err_r:.2f}px  "
                                f"missing={_n_missing_r}(none={_rn_none}+drop={_n_drop_r})  "
                                f"miss_cost={_miss_cost_r:.2f}  "
                                f"ransac_key={_s2_key_r:.2f}{_marker}"
                            )
                            if (_s2_key_r, _rn_none) < (_s2_best_key, _s2_best_none):
                                _s2_best_key   = _s2_key_r
                                _s2_best_blobs = list(_rblobs)
                                _s2_best_none  = _rn_none

                        if _s2_best_blobs != best_hyp_blobs:
                            logger.debug(
                                f"[{self._ctrl} | cam {self._cam}] Proximity stage-2 RANSAC: "
                                f"switched rank-1 -> rank-{next((i+1 for i,c in enumerate(top_candidates) if list(c[2])==_s2_best_blobs), '?')}  "
                                f"ransac_key={_s2_best_key:.2f}  "
                                f"old={_fmt_leds(best_hyp_blobs)}  new={_fmt_leds(_s2_best_blobs)}"
                            )
                            best_hyp_blobs = _s2_best_blobs
                            best_none      = _s2_best_none
                        else:
                            logger.debug(
                                f"[{self._ctrl} | cam {self._cam}] Proximity stage-2 RANSAC: "
                                f"rank-1 confirmed  ransac_key={_s2_best_key:.2f}"
                            )
                        _t_stage2 += time.perf_counter() - _t_stage2_start

            # Step 5: build pairs from truly-locked + best hypothesis assignment (skip None).
            locked_assignment = [(int(candidates[k][0]), int(vis_ids[k])) for k in truly_locked_k]
            hyp_assignment    = [(int(best_hyp_blobs[i]), int(vis_ids[hyp_k[i]]))
                                 for i in range(len(best_hyp_blobs)) if best_hyp_blobs[i] is not None]
            for blob_c, led_id in locked_assignment + hyp_assignment:
                pairs.append((blob_c, led_id))
                locked_obj.append(self.model.positions[led_id])
                locked_img.append(blobs[blob_c])

        logger.debug(
            f"[{self._ctrl} | cam {self._cam}] Proximity: {len(pairs)}/{len(blobs)} blobs matched "
            f"of {n_model_visible} visible LEDs"
        )

        if len(pairs) < self._c_prox_min_inliers:
            logger.debug(f"[{self._ctrl} | cam {self._cam}] Proximity: too few pairs ({len(pairs)} < {self._c_prox_min_inliers}) → None")
            return None

        lo = np.array(locked_obj, dtype=np.float32)
        li = np.array(locked_img, dtype=np.float32)

        ok, rvec, tvec, ransac_idx = _ransac_pnp(
            lo, li, K, dc, rvec_pred, tvec_pred,
            reprojection_px=self._c_prox_reproj_px,
            is_fisheye=self.camera.is_fisheye,
        )

        if not ok or ransac_idx is None:
            logger.debug(f"[{self._ctrl} | cam {self._cam}] Proximity: RANSAC failed → None")
            return None

        # DEBUG: compare predicted / locked-only / final poses to gauge whether
        # refining the predicted pose from locked pairs (for candidate tightening)
        # would meaningfully change the result.
        _lk_str = (f"rvec={_locked_pose_dbg[0]} tvec={_locked_pose_dbg[1]}"
                   if _locked_pose_dbg is not None else "n/a")
        # logger.debug(
        #     f"[{self._ctrl} | cam {self._cam}] Pose compare — "
        #     f"\npred: rvec={rvec_pred.ravel()} tvec={tvec_pred.ravel()}  |  "
        #     f"\nlocked-PnP: {_lk_str}  |  "
        #     f"\nfinal: rvec={np.asarray(rvec).ravel()} tvec={np.asarray(tvec).ravel()}"
        # )

        final_pairs   = [pairs[k] for k in ransac_idx]
        inlier_set    = set(ransac_idx)
        dropped_pairs = [pairs[k] for k in range(len(pairs)) if k not in inlier_set]
        if dropped_pairs:
            logger.debug(
                f"[{self._ctrl} | cam {self._cam}] Proximity RANSAC dropped "
                f"{len(dropped_pairs)} pair(s): "
                + "  ".join(f"led={lid} blob={bid}" for bid, lid in dropped_pairs)
            )
        if len(final_pairs) < self._c_prox_min_inliers:
            logger.debug(f"[{self._ctrl} | cam {self._cam}] Proximity: RANSAC too few inliers ({len(final_pairs)}) → None")
            return None

        # Strict visibility re-check with the RANSAC-refined pose (no occlusion margin).
        # Removes LEDs that were included via the permissive predicted-pose check but are
        # actually occluded under the now-accurate pose.
        _R_ref, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32).reshape(3, 1))
        _tvec_ref  = np.asarray(tvec, dtype=np.float32).reshape(3)
        _vis_ref = _visible_mask(
            _R_ref, _tvec_ref,
            self.model.positions, self.model.normals, geom,
            cam_K=K, cam_dc=dc, cam_w=self.camera.width, cam_h=self.camera.height,
            cam_rpmax=self.camera.rpmax, cam_is_fisheye=self.camera.is_fisheye,
            facing_threshold_deg=self._c_facing_deg,
            occlusion_margin_m=0.0,
        )
        if occluders_per_cam:
            _occ_ref = occluders_per_cam.get(self.camera.camera_idx)
            if _occ_ref is not None:
                _R_occ_ref, _t_occ_ref, _geom_occ_ref = _occ_ref
                _cross_occ_ref = _cross_occluded_mask(
                    _R_ref, _tvec_ref, self.model.positions,
                    _R_occ_ref, _t_occ_ref, _geom_occ_ref,
                    self._c_occ_radius, self._c_occ_radius, focal_px, self._c_occ_margin_px,
                    log_tag=f"[{self._ctrl} | cam {self._cam}]",
                    vis_mask=_vis_ref > 0.0,
                )
                _vis_ref[_cross_occ_ref] = 0.0
        _dropped_vis = [(b, l) for b, l in final_pairs if _vis_ref[l] < 1.0]
        if _dropped_vis:
            logger.debug(
                f"[{self._ctrl} | cam {self._cam}] Proximity post-RANSAC vis-drop "
                f"{len(_dropped_vis)} pair(s): " +
                "  ".join(f"led={l} blob={b}" for b, l in _dropped_vis)
            )
            final_pairs = [(b, l) for b, l in final_pairs if _vis_ref[l] >= 1.0]
            if len(final_pairs) < self._c_prox_min_inliers:
                logger.debug(f"[{self._ctrl} | cam {self._cam}] Proximity: post-vis-drop too few pairs → None")
                return None

        lo_f = self.model.positions[[l for _, l in final_pairs]].astype(np.float32)
        li_f = blobs[[b for b, _ in final_pairs]].astype(np.float32)
        pe   = np.linalg.norm(_project_points(rvec, tvec, lo_f, K, dc,
                                               is_fisheye=self.camera.is_fisheye) - li_f, axis=1)
        error     = float(pe.mean())
        max_error = float(pe.max())

        # RANSAC-confirmed inlier LEDs used as anchors for aux camera snapping.
        final_lids = [lid for _, lid in final_pairs]
        final_obj  = self.model.positions[final_lids].astype(np.float32)

        aux_snapped_per_cam: Dict[int, List] = {}

        # Aux cameras: Hungarian over all model-visible LEDs × all aux blobs.
        # Same dummy-column pattern as the primary path prevents LEDs without a
        # nearby blob from stealing real blobs from LEDs that have one.
        _t_aux_start = time.perf_counter()
        if other_cameras_blobs:
            _R_ref, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32).reshape(3, 1))
            _T_world_ref = self.T_world_cam.compose(Transform(_R_ref, np.asarray(tvec, dtype=np.float32).reshape(3)))
            for _ocam, _oblobs, _ in other_cameras_blobs:
                _oblobs = np.asarray(_oblobs, dtype=np.float32)
                _pairs_i: List = []
                if len(_oblobs) > 0:
                    _R_i_r, _t_i_r, _rv_i_r, _focal_i_r, _vis_i_r = self._aux_cam_vis(
                        _ocam, _T_world_ref, occluders_per_cam
                    )

                    _vis_ids_i  = np.where(_vis_i_r)[0]
                    if len(_vis_ids_i) > 0:
                        _vis_pos_i  = self.model.positions[_vis_ids_i].astype(np.float32)
                        _proj_i_r   = _project_points(_rv_i_r, _t_i_r, _vis_pos_i,
                                                      _ocam.camera_matrix, _ocam.dist_coeffs,
                                                      is_fisheye=_ocam.is_fisheye)

                        _cost_i_r = cdist(_proj_i_r, _oblobs)

                        _n_vis_i   = len(_vis_ids_i)
                        _cost_i_aug = np.hstack([_cost_i_r,
                                                 np.full((_n_vis_i, _n_vis_i), self._c_joint_prefilter_px - 1e-6)])
                        _row_r, _col_r = linear_sum_assignment(_cost_i_aug)
                        for _rr, _cc in zip(_row_r, _col_r):
                            if _cc < len(_oblobs) and _cost_i_r[_rr, _cc] < self._c_joint_prefilter_px:
                                _pairs_i.append((int(_cc), int(_vis_ids_i[_rr])))

                aux_snapped_per_cam[_ocam.camera_idx] = _pairs_i
                if is_deep() and _pairs_i:
                    logger.debug(
                        f"[{self._ctrl} | cam {self._cam}] Proximity aux snap cam{_ocam.camera_idx}: "
                        f"{len(_pairs_i)} pairs (refined pose)"
                    )

        aux_inlier_count = 0
        aux_cameras_result: List = []
        for _aux_cam_idx, _aux_pairs_i in aux_snapped_per_cam.items():
            _n_aux = len(_aux_pairs_i)
            aux_inlier_count += _n_aux
            aux_cameras_result.append((_aux_cam_idx, _n_aux))

        _t_aux = time.perf_counter() - _t_aux_start

        # Joint LM refinement: optimise T_world_ctrl over all cameras simultaneously.
        _t_lm_start = time.perf_counter()
        _joint_result = self._apply_joint_lm(
            rvec, tvec, final_pairs, blobs,
            aux_snapped_per_cam, other_cameras_blobs or [],
            K, dc, "Proximity", error,
        )
        if _joint_result is not None:
            rvec, tvec, error = _joint_result
        _t_lm = time.perf_counter() - _t_lm_start

        _aux_log = ""
        if aux_cameras_result:
            _aux_log = "  aux=[" + ",".join(f"cam{c}:{n}" for c, n in aux_cameras_result if n > 0) + "]"
        logger.debug(
            f"[{self._ctrl} | cam {self._cam}] Proximity: OK  inliers={len(final_pairs)}  err={error:.2f}px  max={max_error:.2f}px{_aux_log}"
            f"  bench: coll={_t_collisions*1000:.1f}ms  score={_t_scoring*1000:.1f}ms  s2={_t_stage2*1000:.1f}ms  aux={_t_aux*1000:.1f}ms  lm={_t_lm*1000:.1f}ms"
        )
        return {
            "rvec":       rvec,
            "tvec":       tvec,
            "error":      error,
            "assignment": final_pairs,
            "method":          "proximity_hungarian",
            "aux_inliers":     aux_inlier_count,
            "aux_cameras":     aux_cameras_result or None,
            "aux_assignments": dict(aux_snapped_per_cam) if aux_snapped_per_cam else None,
        }

    def constrained_search(
        self,
        blobs: np.ndarray,
        predicted_pose: Tuple[np.ndarray, np.ndarray],
        prior_assignment: Optional[List] = None,
        blob_radii: Optional[np.ndarray] = None,
        other_cameras_blobs: Optional[List] = None,
    ) -> Optional[Dict]:
        """
        Prior-constrained translation solver for 2–3 blob frames.

        When only 2–3 blobs are visible, P3P/RANSAC cannot run.  This solver
        fixes the rotation from the predicted pose as a hard constraint and
        collapses the 6-DOF problem to translation-only (3-DOF).

        P2P mode  (n_blobs == 3):
            Project prior LEDs → snap 3 pairs.
            Build a 4×3 overdetermined system from 2 pairs plus any aux-camera
            pairs and solve for (tx, ty, tz) via least-squares.
            Validate with the 3rd pair.

        P1P mode  (n_blobs == 2):
            Project prior LEDs → snap 2 pairs.
            If aux cameras supply ≥1 additional pair the system is overdetermined
            and all three translation components are solved freely.
            Otherwise fix tz = tvec_prior[2] (depth cannot be recovered from one
            primary correspondence alone) and solve (tx, ty) analytically.
            Validate with the 2nd primary pair.

        All primary pairs must reproject within reprojection_threshold after
        solving — there are too few primary matches to average away a bad one.

        Multi-camera notes:
            Aux pairs are pre-snapped with the predicted pose, contribute linear
            rows to the same translation solve, then re-snapped with the solved
            pose for the output aux_assignments dict.
            The linear equation for a pair in aux camera i with relative transform
            (A, c) = T_{aux←primary} is:
                q = A @ R_prior @ P_ctrl + c   (known)
                (A[0] − xn·A[2]) @ u = xn·q[2] − q[0]
                (A[1] − yn·A[2]) @ u = yn·q[2] − q[1]
            Primary rows are the A=I, c=0 special case.
        """
        blobs   = np.asarray(blobs, dtype=np.float32)
        n_blobs = len(blobs)

        if n_blobs < 2:
            return None
        if prior_assignment is None or len(prior_assignment) < 2:
            return None

        rvec_pred, tvec_pred = predicted_pose
        rvec_pred  = np.asarray(rvec_pred, dtype=np.float32).reshape(3, 1)
        tvec_pred  = np.asarray(tvec_pred, dtype=np.float32).reshape(3)
        R_prior, _ = cv2.Rodrigues(rvec_pred)

        K  = self.camera.camera_matrix
        dc = self.camera.dist_coeffs

        mode       = 'p2p' if n_blobs >= 3 else 'p1p'
        n_required = 3     if mode == 'p2p' else 2

        # ── Primary snap ──────────────────────────────────────────────────────────
        prior_lids = [lid for _, lid in prior_assignment]
        prior_obj  = self.model.positions[prior_lids].astype(np.float32)
        proj_prior = _project_points(rvec_pred, tvec_pred, prior_obj, K, dc,
                                     is_fisheye=self.camera.is_fisheye)
        led_cam    = (R_prior @ prior_obj.T).T + tvec_pred
        focal_px   = float(max(K[0, 0], K[1, 1]))

        locked_pairs: List[Tuple[int, int]] = []
        used_blobs: set = set()

        for i, lid in enumerate(prior_lids):
            if len(locked_pairs) >= n_required:
                break

            depth       = float(max(led_cam[i, 2], 0.01))
            expected_px = focal_px * (self._c_led_radius_mm / 1000.0) / depth
            snap_px     = expected_px * self._c_cs_snap_factor

            if blob_radii is not None:
                eligible = (
                    (blob_radii >= expected_px * self._c_blob_size_min) &
                    (blob_radii <= expected_px * self._c_blob_size_max)
                )
            else:
                eligible = None

            dists = np.linalg.norm(blobs - proj_prior[i], axis=1)

            if eligible is not None and eligible.any():
                cand_idx   = np.where(eligible)[0]
                dists_cand = dists[cand_idx]
                size_err   = np.abs(blob_radii[cand_idx] - expected_px)
                scores     = dists_cand + self._c_blob_size_score_w * size_err
                j          = int(cand_idx[np.argmin(scores)])
            else:
                j = int(np.argmin(dists))

            if dists[j] < snap_px and j not in used_blobs:
                locked_pairs.append((j, lid))
                used_blobs.add(j)

        if len(locked_pairs) < n_required:
            logger.debug(
                f"prior_constrained ({mode}): snapped {len(locked_pairs)}/{n_required} pairs → None"
            )
            return None

        pairs_hyp = locked_pairs[:n_required - 1]   # hypothesis pairs (2 for P2P, 1 for P1P)
        pair_val  = locked_pairs[n_required - 1]     # validating pair

        # ── Aux pre-snap with predicted pose (augments the linear system) ─────────
        # Maps cam_idx → (Camera, blobs_array_f32, [(blob_idx, led_id)])
        _aux_pre: Dict = {}
        if other_cameras_blobs:
            _T_world_pred = self.T_world_cam.compose(
                Transform(R_prior.astype(np.float32), tvec_pred)
            )
            for _ocam, _oblobs, _oradii in other_cameras_blobs:
                _obl = np.asarray(_oblobs, dtype=np.float32)
                if len(_obl) == 0:
                    _aux_pre[_ocam.camera_idx] = (_ocam, _obl, [])
                    continue
                _T_ci_p     = _ocam.T_world_cam.inverse().compose(_T_world_pred)
                _R_ci_p     = _T_ci_p.R.astype(np.float32)
                _t_ci_p     = _T_ci_p.t.astype(np.float32)
                _rv_ci_p, _ = cv2.Rodrigues(_R_ci_p)
                _proj_ci_p  = _project_points(_rv_ci_p, _t_ci_p, prior_obj,
                                               _ocam.camera_matrix, _ocam.dist_coeffs,
                                               is_fisheye=_ocam.is_fisheye)
                _led_ci_p   = (_R_ci_p @ prior_obj.T).T + _t_ci_p
                _focal_ci   = float(max(_ocam.camera_matrix[0, 0], _ocam.camera_matrix[1, 1]))
                _pairs_ci = self._snap_led_pairs(
                    prior_lids, _proj_ci_p, _led_ci_p, _obl, _oradii, _focal_ci
                )
                _aux_pre[_ocam.camera_idx] = (_ocam, _obl, _pairs_ci)

        n_aux_pre = sum(len(v[2]) for v in _aux_pre.values())

        # ── Undistort primary hypothesis blob positions ───────────────────────────
        hyp_blobs  = np.array([blobs[b] for b, _ in pairs_hyp], dtype=np.float32)
        pts_undist = self.camera.undistort_points(hyp_blobs)

        # ── Translation solve ─────────────────────────────────────────────────────
        # General linear equation per pair in a camera with relative transform
        # (A, c) = T_{cam←primary} applied to the primary-frame pose unknowns u:
        #   P_cam = A @ R_prior @ P_ctrl + A @ u + c  =  q + A @ u
        #   (A[0] − xn·A[2]) @ u = xn·q[2] − q[0]
        #   (A[1] − yn·A[2]) @ u = yn·q[2] − q[1]
        # Primary camera: A = I, c = 0  →  [1,0,−xn] / [0,1,−yn] rows.
        if mode == 'p2p' or n_aux_pre > 0:
            # Enough constraints for unconstrained 3-DOF least-squares
            A_rows: List = []
            b_rows: List = []
            for k, (_, lk) in enumerate(pairs_hyp):
                Pi  = self.model.positions[lk].astype(np.float64)
                xn  = float(pts_undist[k, 0])
                yn  = float(pts_undist[k, 1])
                r0  = float(R_prior[0] @ Pi)
                r1  = float(R_prior[1] @ Pi)
                r2  = float(R_prior[2] @ Pi)
                A_rows += [[1.0, 0.0, -xn], [0.0, 1.0, -yn]]
                b_rows += [xn * r2 - r0,    yn * r2 - r1]
            for _, (_ocam, _obl, _pairs_ci) in _aux_pre.items():
                if not _pairs_ci:
                    continue
                _T_ap  = _ocam.T_world_cam.inverse().compose(self.T_world_cam)
                _A     = _T_ap.R.astype(np.float64)
                _c     = _T_ap.t.astype(np.float64)
                _aux_hyp = np.array([_obl[b_idx] for b_idx, _ in _pairs_ci], dtype=np.float32)
                _und_aux = _ocam.undistort_points(_aux_hyp)
                for _k, (_, _lid) in enumerate(_pairs_ci):
                    _Pi = self.model.positions[_lid].astype(np.float64)
                    _q  = _A @ (R_prior.astype(np.float64) @ _Pi) + _c
                    _xn = float(_und_aux[_k, 0])
                    _yn = float(_und_aux[_k, 1])
                    A_rows.append((_A[0] - _xn * _A[2]).tolist())
                    b_rows.append(float(_xn * _q[2] - _q[0]))
                    A_rows.append((_A[1] - _yn * _A[2]).tolist())
                    b_rows.append(float(_yn * _q[2] - _q[1]))
            t_solved, _, _, _ = np.linalg.lstsq(
                np.array(A_rows, dtype=np.float64),
                np.array(b_rows, dtype=np.float64),
                rcond=None,
            )
        else:
            # P1P with no aux: fix depth from prior
            tz  = float(tvec_pred[2])
            Pi  = self.model.positions[pairs_hyp[0][1]].astype(np.float64)
            xn  = float(pts_undist[0, 0])
            yn  = float(pts_undist[0, 1])
            r0  = float(R_prior[0] @ Pi)
            r1  = float(R_prior[1] @ Pi)
            r2  = float(R_prior[2] @ Pi)
            tx  = xn * (r2 + tz) - r0
            ty  = yn * (r2 + tz) - r1
            t_solved = np.array([tx, ty, tz])

        t_solved = np.asarray(t_solved, dtype=np.float64)

        # Depth sanity check
        if not _check_z_range(t_solved.astype(np.float32)):
            logger.debug(
                f"prior_constrained ({mode}): solved depth {t_solved[2]:.3f} m out of range → None"
            )
            return None

        # ── Validate primary pairs ────────────────────────────────────────────────
        all_primary = pairs_hyp + [pair_val]
        all_obj     = self.model.positions[[l for _, l in all_primary]].astype(np.float32)
        all_img     = blobs[[b for b, _ in all_primary]].astype(np.float32)
        proj_all    = _project_points(rvec_pred, t_solved.astype(np.float32), all_obj, K, dc,
                                      is_fisheye=self.camera.is_fisheye)
        errors      = np.linalg.norm(proj_all - all_img, axis=1)

        if np.any(errors > self._c_cs_reproj_px):
            logger.debug(
                f"prior_constrained ({mode}): validation failed "
                f"errors={errors.round(2)} thresh={self._c_cs_reproj_px} → None"
            )
            return None

        error = float(np.mean(errors))

        # ── Post-solve aux snap with solved pose → aux_assignments ────────────────
        aux_snapped_per_cam: Dict = {}
        if other_cameras_blobs:
            _T_world_solved = self.T_world_cam.compose(
                Transform(R_prior.astype(np.float32), t_solved.astype(np.float32))
            )
            for _ocam, _oblobs, _oradii in other_cameras_blobs:
                _obl = np.asarray(_oblobs, dtype=np.float32)
                if len(_obl) == 0:
                    aux_snapped_per_cam[_ocam.camera_idx] = []
                    continue
                _T_ci_s     = _ocam.T_world_cam.inverse().compose(_T_world_solved)
                _R_ci_s     = _T_ci_s.R.astype(np.float32)
                _t_ci_s     = _T_ci_s.t.astype(np.float32)
                _rv_ci_s, _ = cv2.Rodrigues(_R_ci_s)
                _proj_ci_s  = _project_points(_rv_ci_s, _t_ci_s, prior_obj,
                                               _ocam.camera_matrix, _ocam.dist_coeffs,
                                               is_fisheye=_ocam.is_fisheye)
                _led_ci_s   = (_R_ci_s @ prior_obj.T).T + _t_ci_s
                _focal_ci   = float(max(_ocam.camera_matrix[0, 0], _ocam.camera_matrix[1, 1]))
                aux_snapped_per_cam[_ocam.camera_idx] = self._snap_led_pairs(
                    prior_lids, _proj_ci_s, _led_ci_s, _obl, _oradii, _focal_ci
                )

        aux_inlier_count   = sum(len(v) for v in aux_snapped_per_cam.values())
        aux_cameras_result = [(idx, len(pairs))
                              for idx, pairs in aux_snapped_per_cam.items() if pairs]

        _aux_log = ""
        if aux_cameras_result:
            _aux_log = "  aux=[" + ",".join(f"cam{c}:{n}" for c, n in aux_cameras_result) + "]"
        logger.debug(
            f"[{self._ctrl} | cam {self._cam}] prior_constrained ({mode}): OK  pairs={len(all_primary)}  err={error:.2f}px"
            + (f"  n_aux_solve={n_aux_pre}" if n_aux_pre > 0 else "")
            + _aux_log
        )

        return {
            "rvec":          rvec_pred.astype(np.float64).reshape(3, 1),
            "tvec":          t_solved.reshape(3, 1),
            "error":         error,
            "assignment":    all_primary,
            "method":        f"prior_constrained_{mode}",
            "aux_inliers":   aux_inlier_count,
            "aux_cameras":   aux_cameras_result or None,
            "aux_assignments": dict(aux_snapped_per_cam) if aux_snapped_per_cam else None,
        }

    def brute_search(
        self,
        blobs: np.ndarray,
        depth_tiers: Tuple[Tuple, ...] = ((2, 3), (2, 4), (2, 4, 'edge'), (3, 5), (3, 5, 'edge'), (4, 6)),  # (led_max, blob_max[, 'standard'|'edge'])
        p4_threshold_px: float = 2.0,
        hungarian_threshold_px: float = 5.0,  # pre-filter on the raw P3P hypothesis pose. Loose because P3P poses can be noisy; RANSAC does the real filtering after this
        reprojection_threshold: float = 1.5,  # passed to RANSAC, controls which blobs make it into the final assignment. This is now what the visualization reflects: all shown errors will be ≤ this
        min_inliers: int = 4,
        min_inlier_fraction: Optional[float] = None,
        strong_match_inliers: int = 7,
        strong_match_error_px: float = 1.5,
        min_vis_coverage: float = 0.75,
        pose_prior: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        rng_seed: Optional[int] = 42,
        other_cameras_blobs: Optional[List] = None,
        blob_radii: Optional[np.ndarray] = None,
        blob_mask: Optional[np.ndarray] = None,
        occluders_per_cam: Optional[Dict[int, Tuple[np.ndarray, np.ndarray, object]]] = None,
    ) -> Optional[Dict]:
        """
        Exhaustive pose search via P3P over LED/blob triple correspondences.
        Follows OpenHMD's correspondence_search structure:
          - Outer loop: LED triple (anchor + 2 neighbours, precomputed; no duplicates)
          - Middle loop: every blob as a potential blob anchor
          - Inner loop: C(k,2) blob pairs × 2 orderings (covers all LED–blob assignments)

        Each unique (anchor, l1, l2) ↔ (b_anchor, b1, b2) bijection is evaluated exactly once.
        Gate check: any remaining gate LED projecting near any remaining gate blob.

        Progressive 2D deepening via depth_tiers — each (led_max, blob_max) pair defines a tier.
        Each tier evaluates only the (LED triple, blob pair) combinations not covered by prior tiers:
          - LED triples newly eligible because depth ≤ led_max but depth > prev led_max
          - Blob pairs newly eligible because i2 ≥ prev blob_max for already-eligible triples
        Exits on strong match at the end of any tier's LED triple.
        """
        _log_size_filter = is_deep() and self._c_log_size_filter

        blobs   = np.asarray(blobs, dtype=np.float32)
        n_blobs = len(blobs)
        _avail_idx  = np.where(blob_mask)[0].astype(np.int32) if blob_mask is not None else np.arange(n_blobs, dtype=np.int32)
        n_available = len(_avail_idx)
        if n_available < 4:
            return None

        if self._c_brute_min_frac is not None:
            fraction_floor     = int(np.ceil(self._c_brute_min_frac * n_available))
            min_inliers_eff    = max(self._c_brute_min_inliers, fraction_floor)
            strong_inliers_eff = min(self._c_brute_strong_in, fraction_floor)
        else:
            min_inliers_eff    = self._c_brute_min_inliers
            strong_inliers_eff = self._c_brute_strong_in

        positions = self.model.positions.astype(np.float32)
        normals   = self.model.normals.astype(np.float32)
        K         = self.camera.camera_matrix
        dc        = self.camera.dist_coeffs
        focal_px  = float(max(K[0, 0], K[1, 1]))

        led_triple_idx   = self._led_triple_idx    # (N_LT, 3) int32
        led_triple_depth = self._led_triple_depth  # (N_LT,) int32
        led_triple_gates = self._led_triple_gates  # List[np.ndarray] gate LED indices per triple

        led_triple_idx_edge   = self._led_triple_idx_edge
        led_triple_depth_edge = self._led_triple_depth_edge
        led_triple_gates_edge = self._led_triple_gates_edge

        geom = self._geometry

        max_blob_depth = max(t[1] for t in self._c_brute_depth_tiers)
        blob_nbr = _build_blob_neighbor_lists(blobs, k=max_blob_depth)

        # Undistort blobs once (mirrors OpenHMD's correspondence_search_set_blobs).
        # Gate check uses pinhole projection which is valid in undistorted space.
        blobs_undist = self.camera.undistort_points(blobs, P=K).astype(np.float32)

        p4_thresh_sq = self._c_brute_p4_px ** 2
        fx, fy       = float(K[0, 0]), float(K[1, 1])
        cx, cy       = float(K[0, 2]), float(K[1, 2])

        R_prior    = None
        tvec_prior = None
        if pose_prior is not None:
            rvec_pr, tvec_pr = pose_prior
            R_prior, _ = cv2.Rodrigues(np.asarray(rvec_pr, dtype=np.float32).reshape(3, 1))
            tvec_prior = np.asarray(tvec_pr, dtype=np.float32).reshape(3)

        best_solution      = None
        best_inliers       = 0
        best_inliers_total = 0
        best_error         = np.inf
        best_orient_err    = np.inf
        best_tvec_err      = np.inf
        strong_found       = False
        solution_tier      = None

        # Per-triple: how far blob depth has been explored (the blob_max of the last tier
        # that processed this triple). Enables delta coverage across tiers.
        # Maintained separately for each neighbourhood type so delta tracking is consistent.
        prev_blob_max_per_triple      = np.zeros(len(led_triple_idx),      dtype=np.int32)
        prev_blob_max_per_triple_edge = np.zeros(len(led_triple_idx_edge), dtype=np.int32)

        rng = np.random.default_rng(self._c_brute_rng_seed)

        _track_gate_dist = is_deep()

        tier_p3p_calls = [0] * len(self._c_brute_depth_tiers)
        tier_lq_tried  = [0] * len(self._c_brute_depth_tiers)
        tier_lq_total  = [0] * len(self._c_brute_depth_tiers)

        _dbg_leds, _dbg_blobs = get_debug_triple()
        debug_active      = _dbg_leds is not None or _dbg_blobs is not None
        debug_led_anchor  = int(_dbg_leds[0])     if _dbg_leds  is not None else None
        debug_led_set     = frozenset(_dbg_leds)  if _dbg_leds  is not None else None
        debug_blob_anchor = int(_dbg_blobs[0])    if _dbg_blobs is not None else None
        debug_blob_set    = frozenset(_dbg_blobs) if _dbg_blobs is not None else None

        seen_bijections:  set                  = set()
        bijection_counts: Dict[frozenset, int] = {} if is_deep() else None

        for tier_idx, tier_spec in enumerate(self._c_brute_depth_tiers):
            if strong_found:
                break

            led_max, blob_max = tier_spec[0], tier_spec[1]
            nbr_type = tier_spec[2] if len(tier_spec) > 2 else 'standard'

            if nbr_type == 'edge':
                cur_triple_idx   = led_triple_idx_edge
                cur_triple_depth = led_triple_depth_edge
                cur_triple_gates = led_triple_gates_edge
                cur_prev_blob    = prev_blob_max_per_triple_edge
            else:
                cur_triple_idx   = led_triple_idx
                cur_triple_depth = led_triple_depth
                cur_triple_gates = led_triple_gates
                cur_prev_blob    = prev_blob_max_per_triple

            eligible_mask = cur_triple_depth <= led_max
            eligible_triple_idx = np.where(eligible_mask)[0]

            # Keep only triples that have new blob pairs to explore at this tier.
            has_new_blob_pairs = cur_prev_blob[eligible_triple_idx] < blob_max
            active_triple_idx  = eligible_triple_idx[has_new_blob_pairs]
            tier_lq_total[tier_idx] = len(active_triple_idx)

            if len(active_triple_idx) == 0:
                continue

            active_triple_idx = active_triple_idx[rng.permutation(len(active_triple_idx))]

            for triple_i in active_triple_idx:
                led_ids            = cur_triple_idx[triple_i]    # [anchor, l1, l2]
                p3p_world_pts      = positions[led_ids]           # (3, 3) world points for P3P
                gate_led           = cur_triple_gates[triple_i]
                gate_led_world_pts = positions[gate_led].astype(np.float32) if len(gate_led) > 0 else np.zeros((0, 3), dtype=np.float32)

                # Start blob-pair enumeration from where the previous tier left off for
                # this triple; avoids re-evaluating combinations already covered earlier.
                min_blob_i2 = int(cur_prev_blob[triple_i])
                did_p3p = False

                for b_anchor in _avail_idx:
                    if strong_found:
                        break
                    blob_neighbors   = blob_nbr[b_anchor]
                    if blob_mask is not None:
                        blob_neighbors = blob_neighbors[blob_mask[blob_neighbors]]
                    n_blob_neighbors = min(len(blob_neighbors), blob_max)
                    if n_blob_neighbors < 2:
                        continue

                    for i1, i2 in combinations(range(n_blob_neighbors), 2):
                        if strong_found:
                            break
                        if i2 < min_blob_i2:
                            continue
                        b1 = int(blob_neighbors[i1])
                        b2 = int(blob_neighbors[i2])

                        gate_blob_idx    = [int(blob_neighbors[j]) for j in range(n_blob_neighbors) if j != i1 and j != i2]
                        # If gate_blob_idx is empty (n_blob_neighbors == 2), _gate_any_point
                        # returns False — a 4th blob neighbour is required for gate validation.
                        gate_blob_img_pts = blobs_undist[gate_blob_idx] if gate_blob_idx else np.zeros((0, 2), dtype=np.float32)

                        for b1_ord, b2_ord in ((b1, b2), (b2, b1)):
                            if strong_found:
                                break

                            # ── Debug trigger ─────────────────────────────────────
                            # Anchors are matched positionally (first element of each
                            # debug triple); l1/l2 and b1/b2 are matched as sets so
                            # their internal ordering doesn't matter.  This gives at
                            # most 2 prints per frame: one per b1↔b2 swap.
                            dbg = is_verbose_all() or (
                                debug_active and
                                (debug_led_set  is None or (
                                    int(led_ids[0]) == debug_led_anchor and
                                    frozenset(led_ids) == debug_led_set)) and
                                (debug_blob_set is None or (
                                    b_anchor == debug_blob_anchor and
                                    frozenset([b_anchor, b1_ord, b2_ord]) == debug_blob_set))
                            )
                            if dbg:
                                logger.debug(
                                    f"Target triple reached — "
                                    f"LEDs {list(led_ids)}  blobs [{b_anchor},{b1_ord},{b2_ord}]  "
                                    f"tier={tier_idx} ({_tier_label(tier_spec)})"
                                )
                                logger.debug(
                                    f"  gate LEDs={list(gate_led)}  gate blobs={gate_blob_idx}"
                                )

                            p3p_img_pts = blobs[[b_anchor, b1_ord, b2_ord]]

                            bij = frozenset(((int(led_ids[0]), b_anchor),
                                             (int(led_ids[1]), b1_ord),
                                             (int(led_ids[2]), b2_ord)))

                            if bij in seen_bijections:
                                continue
                            seen_bijections.add(bij)

                            if bijection_counts is not None:
                                bijection_counts[bij] = bijection_counts.get(bij, 0) + 1

                            # ── 1. P3P → up to 4 pose hypotheses ─────────────────
                            tier_p3p_calls[tier_idx] += 1
                            did_p3p = True
                            # solveP3P only supports standard polynomial distortion
                            # internally. Pre-undistort to normalised coords and pass
                            # identity K so both radtan8 and kb4 work correctly.
                            p3p_img_norm = self.camera.undistort_points(p3p_img_pts)
                            n_sols, rvecs, tvecs = cv2.solveP3P(
                                p3p_world_pts.reshape(3, 1, 3),
                                p3p_img_norm.reshape(3, 1, 2).astype(np.float32),
                                np.eye(3, dtype=np.float32),
                                np.zeros(4, dtype=np.float32),
                                flags=cv2.SOLVEPNP_P3P,
                            )
                            self._dbg(dbg, f"  P3P returned {n_sols} solutions")
                            if not n_sols or rvecs is None:
                                continue

                            # Sort hypotheses by rotation distance to prior so the
                            # closest-to-prior solution is tried first — increases the
                            # chance of strong_found triggering early.
                            if R_prior is not None and n_sols > 1:
                                def _rot_score(rv):
                                    R_i, _ = cv2.Rodrigues(rv.reshape(3, 1).astype(np.float32))
                                    return float(np.trace(R_i @ R_prior.T))
                                order  = sorted(range(n_sols), key=lambda k: -_rot_score(rvecs[k]))
                                rvecs  = [rvecs[k] for k in order]
                                tvecs  = [tvecs[k] for k in order]

                            for sol_i, (rvec_h, tvec_h) in enumerate(zip(rvecs, tvecs)):
                                if strong_found:
                                    break
                                rvec_h = rvec_h.reshape(3, 1).astype(np.float32)
                                tvec_h = tvec_h.reshape(3).astype(np.float32)

                                # ── 2. Depth range check (OpenHMD: 0.05 m – 15 m) ─
                                z_ok = _check_z_range(tvec_h)
                                self._dbg(dbg, f"  sol {sol_i}: z={tvec_h[2]:.3f} m  depth_ok={z_ok}")
                                if not z_ok:
                                    continue

                                R_h, _ = cv2.Rodrigues(rvec_h)

                                # ── 3. Gate check (any gate LED near any gate blob) ─
                                gate_ok, gate_dist = _gate_any_point(R_h, tvec_h, gate_led_world_pts, gate_blob_img_pts, fx, fy, cx, cy, p4_thresh_sq, _track_gate_dist)
                                self._dbg(dbg, f"  sol {sol_i}: gate_ok={gate_ok}, dist={gate_dist:.2f}px")
                                if not gate_ok:
                                    continue

                                # ── 4. Full inlier count on all visible LEDs ───────
                                vis_mask_h = _visible_mask(
                                    R_h, tvec_h, positions, normals,
                                    geom,
                                    cam_K=K, cam_dc=dc, cam_w=self.camera.width, cam_h=self.camera.height,
                                    cam_rpmax=self.camera.rpmax, cam_is_fisheye=self.camera.is_fisheye,
                                    facing_threshold_deg=self._c_facing_deg,
                                ) >= 1.0
                                vis_ids = np.where(vis_mask_h)[0]
                                self._dbg(dbg, f"  sol {sol_i}: {len(vis_ids)} visible LEDs")
                                if len(vis_ids) < self._c_brute_min_inliers:
                                    continue

                                proj_all = _project_points(rvec_h, tvec_h, positions[vis_ids], K, dc,
                                                           is_fisheye=self.camera.is_fisheye)
                                cost     = cdist(blobs, proj_all)
                                if blob_mask is not None:
                                    _cost_sub = cost[_avail_idx]
                                    _sub_rows, hungarian_led_cols = linear_sum_assignment(_cost_sub)
                                    hungarian_blob_rows = _avail_idx[_sub_rows]
                                else:
                                    hungarian_blob_rows, hungarian_led_cols = linear_sum_assignment(cost)

                                inlier_mask    = cost[hungarian_blob_rows, hungarian_led_cols] < self._c_brute_hungarian_px
                                inlier_blobs   = hungarian_blob_rows[inlier_mask]
                                inlier_leds    = vis_ids[hungarian_led_cols[inlier_mask]]

                                if dbg:
                                    outlier_mask     = cost[hungarian_blob_rows, hungarian_led_cols] >= self._c_brute_hungarian_px
                                    outlier_blob_rows = hungarian_blob_rows[outlier_mask]
                                    outlier_led_cols  = vis_ids[hungarian_led_cols[outlier_mask]]
                                    logger.debug(f"  sol {sol_i}: {len(inlier_blobs)} inliers after Hungarian "
                                                 f"(need {min_inliers_eff})")
                                if len(inlier_blobs) < min_inliers_eff:
                                    continue

                                # ── 5. RANSAC PnP refinement on inliers ───────────
                                ok_r, rvec_r, tvec_r, ransac_inliers = _ransac_pnp(
                                    positions[inlier_leds], blobs[inlier_blobs], K, dc,
                                    rvec_h, tvec_h.reshape(3, 1),
                                    reprojection_px=self._c_brute_reproj_px,
                                    is_fisheye=self.camera.is_fisheye,
                                )
                                self._dbg(dbg, f"  sol {sol_i}: RANSAC ok={ok_r}, "
                                               f"inliers={len(ransac_inliers) if ok_r else 0}")
                                if not ok_r:
                                    continue

                                inlier_leds  = inlier_leds[ransac_inliers]
                                inlier_blobs = inlier_blobs[ransac_inliers]

                                if len(inlier_blobs) < min_inliers_eff:
                                    continue

                                # ── 6. Visibility recheck with the refined pose ────
                                # Recompute on ALL LEDs so that:
                                #   (a) the denominator for coverage is accurate,
                                #   (b) inliers occluded under the refined pose are dropped.
                                R_r, _ = cv2.Rodrigues(rvec_r.reshape(3, 1).astype(np.float32))
                                tvec_r_flat = tvec_r.reshape(3)
                                vis_scores_r = _visible_mask(
                                    R_r, tvec_r_flat, positions, normals,
                                    geom,
                                    cam_K=K, cam_dc=dc, cam_w=self.camera.width, cam_h=self.camera.height,
                                    cam_rpmax=self.camera.rpmax, cam_is_fisheye=self.camera.is_fisheye,
                                    facing_threshold_deg=self._c_facing_deg,
                                )
                                if occluders_per_cam:
                                    _occ_r = occluders_per_cam.get(self.camera.camera_idx)
                                    if _occ_r is not None:
                                        _R_occ_r, _t_occ_r, _geom_occ_r = _occ_r
                                        _cross_occ_r = _cross_occluded_mask(
                                            R_r, tvec_r_flat, positions,
                                            _R_occ_r, _t_occ_r, _geom_occ_r,
                                            self._c_occ_radius, self._c_occ_radius, focal_px, self._c_occ_margin_px,
                                            log_tag=f"[{self._ctrl} | cam {self._cam}]",
                                            vis_mask=vis_scores_r > 0.0,
                                        )
                                        vis_scores_r[_cross_occ_r] = 0.0
                                vis_mask_r = vis_scores_r >= 1.0
                                # Drop inliers that became occluded under the refined pose.
                                inlier_still_visible = vis_mask_r[inlier_leds]
                                inlier_leds  = inlier_leds[inlier_still_visible]
                                inlier_blobs = inlier_blobs[inlier_still_visible]
                                self._dbg(dbg, f"  sol {sol_i}: {len(inlier_blobs)} inliers after vis recheck")
                                if len(inlier_blobs) < min_inliers_eff:
                                    continue

                                vis_ids_r = np.where(vis_mask_r)[0]

                                # ── 6.5. Post-RANSAC blob recovery ────────────────
                                # Blobs outside hungarian_threshold_px on the coarse
                                # P3P pose may land within reprojection_threshold
                                # under the refined pose.  One greedy nearest-
                                # neighbour pass recovers them (one cdist, no PnP).
                                matched_blob_set  = set(inlier_blobs.tolist())
                                matched_led_set   = set(inlier_leds.tolist())
                                unmatched_blobs   = np.array([b for b in _avail_idx if b not in matched_blob_set], dtype=np.int32)
                                unmatched_col_idx = np.array([j for j, lid in enumerate(vis_ids_r) if int(lid) not in matched_led_set], dtype=np.int32)

                                if len(unmatched_blobs) > 0 and len(unmatched_col_idx) > 0:
                                    proj_vis_r = _project_points(rvec_r, tvec_r, positions[vis_ids_r], K, dc,
                                                                  is_fisheye=self.camera.is_fisheye)
                                    cost_r     = cdist(blobs, proj_vis_r)
                                    sub_min    = cost_r[np.ix_(unmatched_blobs, unmatched_col_idx)].min(axis=0)
                                    _led_cam_r = ((R_r @ positions[vis_ids_r].T).T + tvec_r_flat
                                                  if blob_radii is not None else None)
                                    extra_blobs: List[int] = []
                                    extra_leds:  List[int] = []
                                    for order_j in np.argsort(sub_min):
                                        if sub_min[order_j] >= self._c_brute_reproj_px:
                                            break
                                        col    = int(unmatched_col_idx[order_j])
                                        led_id = int(vis_ids_r[col])
                                        _exp_px_rc = (focal_px * (self._c_led_radius_mm / 1000.0) /
                                                      float(max(_led_cam_r[col, 2], 0.01))
                                                      if _led_cam_r is not None else None)
                                        for row_i in np.argsort(cost_r[unmatched_blobs, col]):
                                            b = int(unmatched_blobs[row_i])
                                            if b in matched_blob_set:
                                                continue
                                            if cost_r[b, col] >= self._c_brute_reproj_px:
                                                break
                                            if (_exp_px_rc is not None and not (
                                                    _exp_px_rc * self._c_blob_size_min
                                                    <= float(blob_radii[b])
                                                    <= _exp_px_rc * self._c_blob_size_max)):
                                                if _log_size_filter:
                                                    logger.debug(
                                                        f"  LED {led_id}: blob {b}"
                                                        f" size-filtered (brute extra-blob)"
                                                        f"  r={float(blob_radii[b]):.2f}  expected"
                                                        f" {_exp_px_rc*self._c_blob_size_min:.2f}–{_exp_px_rc*self._c_blob_size_max:.2f}px"
                                                    )
                                                continue
                                            matched_blob_set.add(b)
                                            extra_blobs.append(b)
                                            extra_leds.append(led_id)
                                            break
                                    if extra_blobs:
                                        inlier_blobs = np.concatenate([inlier_blobs, np.array(extra_blobs, dtype=inlier_blobs.dtype)])
                                        inlier_leds  = np.concatenate([inlier_leds,  np.array(extra_leds,  dtype=inlier_leds.dtype)])
                                        self._dbg(dbg, f"  sol {sol_i}: +{len(extra_blobs)} blob(s) recovered post-RANSAC")

                                # ── 6.7. Aux-camera validation ────────────────────
                                # Lift refined pose to world frame and score against
                                # each aux camera's blobs. These correspondences
                                # are not added to the primary-cam assignment but do
                                # increase the pooled coverage and inlier count used
                                # for hypothesis ranking.
                                extra_vis_weight      = 0.0
                                extra_inlier_weight   = 0.0
                                extra_inlier_count    = 0
                                aux_blob_denom        = 0
                                aux_cameras_current   = []
                                aux_assignments_current: Dict[int, List] = {}
                                if other_cameras_blobs:
                                    T_world_ctrl = self.T_world_cam.compose(
                                        Transform(R_r, tvec_r_flat)
                                    )
                                    for _ocam, _oblobs, _oradii in other_cameras_blobs:
                                        _oblobs = np.asarray(_oblobs, dtype=np.float32)
                                        if len(_oblobs) == 0:
                                            continue
                                        _R_i, _t_i, _rv_i, _focal_i, _vis_i = self._aux_cam_vis(
                                            _ocam, T_world_ctrl, occluders_per_cam
                                        )
                                        _vis_ids_i = np.where(_vis_i)[0]
                                        if len(_vis_ids_i) == 0:
                                            continue

                                        _proj_i = _project_points(
                                            _rv_i, _t_i, positions[_vis_ids_i],
                                            _ocam.camera_matrix, _ocam.dist_coeffs,
                                            is_fisheye=_ocam.is_fisheye,
                                        )
                                        _cost_i = cdist(_oblobs, _proj_i)
                                        _rows_i, _cols_i = linear_sum_assignment(_cost_i)
                                        _inlier_i = _cost_i[_rows_i, _cols_i] < self._c_brute_aux_reproj_px
                                        _led_cam_i = (_R_i @ positions[_vis_ids_i].T).T + _t_i
                                        if _oradii is not None:
                                            for _k in range(len(_rows_i)):
                                                if not _inlier_i[_k]:
                                                    continue
                                                _depth_k = float(max(_led_cam_i[_cols_i[_k], 2], 0.01))
                                                _exp_px_k = _focal_i * (self._c_led_radius_mm / 1000.0) / _depth_k
                                                if not (_exp_px_k * self._c_blob_size_min
                                                        <= float(_oradii[_rows_i[_k]])
                                                        <= _exp_px_k * self._c_blob_size_max):
                                                    if _log_size_filter:
                                                        logger.debug(
                                                            f"  LED {int(_vis_ids_i[_cols_i[_k]])}: blob {int(_rows_i[_k])}"
                                                            f" size-filtered (brute aux-inlier cam{_ocam.camera_idx})"
                                                            f"  r={float(_oradii[_rows_i[_k]]):.2f}  expected"
                                                            f" {_exp_px_k*self._c_blob_size_min:.2f}–{_exp_px_k*self._c_blob_size_max:.2f}px"
                                                        )
                                                    _inlier_i[_k] = False
                                        _n_aux = int(_inlier_i.sum())
                                        if dbg:
                                            _matched_dists = _cost_i[_rows_i, _cols_i]
                                            logger.debug(
                                                f"  sol {sol_i}: aux cam{_ocam.camera_idx} "
                                                f"vis={len(_vis_ids_i)} blobs={len(_oblobs)} "
                                                f"matched_dists={_matched_dists.round(1).tolist()} "
                                                f"thresh={self._c_brute_aux_reproj_px:.1f}px → {_n_aux} inliers"
                                            )
                                        extra_inlier_count += _n_aux
                                        aux_blob_denom     += min(len(_oblobs), len(_vis_ids_i))
                                        aux_cameras_current.append((_ocam.camera_idx, _n_aux))
                                        aux_assignments_current[_ocam.camera_idx] = [
                                            (int(_rows_i[_k]), int(_vis_ids_i[_cols_i[_k]]))
                                            for _k in range(len(_rows_i)) if _inlier_i[_k]
                                        ]

                                        _led_nrm_i = (_R_i @ normals[_vis_ids_i].T).T
                                        _vdirs_i   = -_led_cam_i / (np.linalg.norm(_led_cam_i, axis=1, keepdims=True) + 1e-9)
                                        _w_i       = np.clip((_led_nrm_i * _vdirs_i).sum(axis=1), 0.0, 1.0)

                                        extra_vis_weight    += float(_w_i.sum())
                                        extra_inlier_weight += float(_w_i[_cols_i[_inlier_i]].sum())

                                # ── 7. Visibility coverage check ──────────────────
                                # Weight each visible LED by cos(θ) — the dot product
                                # of its normal with the view direction.  LEDs at
                                # grazing angles (θ → 90°) may be theoretically
                                # visible but are rarely detected reliably, so they
                                # contribute less to both numerator and denominator.
                                # This prevents those LEDs from unfairly failing the
                                # coverage gate when they simply aren't bright enough.
                                # Counts from other cameras (extra_*) are pooled in so
                                # that combined multi-camera evidence can satisfy the
                                # threshold even when one camera sees few LEDs.
                                n_visible_leds = len(vis_ids_r)
                                n_inlier_blobs = len(inlier_blobs)
                                n_inlier_total = n_inlier_blobs + extra_inlier_count

                                led_cam_pts    = (R_r @ positions[vis_ids_r].T).T + tvec_r_flat
                                led_cam_normals = (R_r @ normals[vis_ids_r].T).T
                                led_view_dirs  = -led_cam_pts / (np.linalg.norm(led_cam_pts, axis=1, keepdims=True) + 1e-9)
                                led_vis_weights = np.clip((led_cam_normals * led_view_dirs).sum(axis=1), 0.0, 1.0)

                                # inlier_leds ⊂ vis_ids_r is guaranteed by step 6; searchsorted
                                # maps each inlier LED index to its position in vis_ids_r so we
                                # can index led_vis_weights without a full boolean mask.
                                inlier_idx_in_vis    = np.searchsorted(vis_ids_r, inlier_leds)
                                weighted_visible_count = float(led_vis_weights.sum()) + extra_vis_weight
                                weighted_inlier_count  = float(led_vis_weights[inlier_idx_in_vis].sum()) + extra_inlier_weight

                                # Balanced F1-style coverage: harmonic mean of LED recall and blob
                                # precision. LED recall (weighted by cos θ) measures how many of the
                                # model-visible LEDs were matched. Blob precision measures how many of
                                # the detected blobs the pose explains, with the denominator capped at
                                # n_visible_leds so that blobs from the other controller or noise do
                                # not unfairly penalise a correct pose. Both sides pool aux cameras.
                                led_cov  = (weighted_inlier_count / weighted_visible_count
                                            if weighted_visible_count > 0 else 1.0)
                                _blob_denom = min(n_available, n_visible_leds) + aux_blob_denom
                                blob_cov = ((n_inlier_blobs + extra_inlier_count) / _blob_denom
                                            if _blob_denom > 0 else 1.0)
                                balanced_coverage = (2.0 * led_cov * blob_cov / (led_cov + blob_cov)
                                                     if led_cov + blob_cov > 0.0 else 0.0)
                                if balanced_coverage < self._c_brute_min_vis_cov:
                                    if dbg:
                                        logger.debug(
                                            f"  sol {sol_i}: balanced coverage {balanced_coverage:.2f} < {self._c_brute_min_vis_cov:.2f}"
                                            f"  led_cov={led_cov:.2f}  blob_cov={blob_cov:.2f}"
                                            f"  (primary {n_inlier_blobs}/{n_visible_leds} leds,"
                                            f" {n_inlier_blobs}/{min(n_available, n_visible_leds)} blobs"
                                            f" +{extra_inlier_count} aux)"
                                        )
                                    continue

                                proj_r = _project_points(rvec_r, tvec_r, positions[inlier_leds], K, dc,
                                                         is_fisheye=self.camera.is_fisheye)
                                err    = float(np.mean(np.linalg.norm(proj_r - blobs[inlier_blobs], axis=1)))

                                orient_err = np.inf
                                if R_prior is not None:
                                    cos_orient_angle = np.clip((np.trace(R_r @ R_prior.T) - 1.0) / 2.0, -1.0, 1.0)
                                    orient_err = float(np.arccos(cos_orient_angle))

                                tvec_err = np.inf
                                if tvec_prior is not None:
                                    tvec_err = float(np.linalg.norm(tvec_r.reshape(3) - tvec_prior))

                                # Prefer more inliers (pooled across all cameras); break
                                # ties by primary error, then prior distances.
                                # The error cap prevents a high-inlier solution at poor
                                # accuracy from displacing a tight low-inlier one.
                                is_better = (
                                    (n_inlier_total > best_inliers_total and err < best_error + 1.0) or
                                    (n_inlier_total >= best_inliers_total + 2 and err < best_error + 1.5) or
                                    (n_inlier_total == best_inliers_total and err < best_error) or
                                    (n_inlier_total == best_inliers_total and
                                     abs(err - best_error) < 0.5 and
                                     orient_err < best_orient_err) or
                                    (n_inlier_total == best_inliers_total and
                                     abs(err - best_error) < 0.5 and
                                     orient_err == best_orient_err and
                                     tvec_err < best_tvec_err)
                                )

                                self._dbg(dbg, f"  sol {sol_i}: err={err:.3f} px  "
                                               f"inliers={n_inlier_blobs}+{extra_inlier_count}aux={n_inlier_total}  "
                                               f"is_better={is_better}")

                                if is_better:
                                    best_solution = {
                                        "rvec":             rvec_r,
                                        "tvec":             tvec_r,
                                        "inliers":          n_inlier_blobs,
                                        "aux_inliers":      extra_inlier_count,
                                        "aux_cameras":      aux_cameras_current or None,
                                        "aux_assignments":  dict(aux_assignments_current) or None,
                                        "error":            err,
                                        "assignment":       list(zip(inlier_blobs.tolist(), inlier_leds.tolist())),
                                        "method":           "p3p_systematic",
                                    }
                                    best_inliers       = n_inlier_blobs
                                    best_inliers_total = n_inlier_total
                                    best_error         = err
                                    best_orient_err    = orient_err
                                    best_tvec_err      = tvec_err
                                    solution_tier      = tier_idx

                                    if log_best():
                                        _aux_dbg = ""
                                        if aux_cameras_current:
                                            _aux_dbg = "  aux=[" + ",".join(
                                                f"cam{c}:{n}" for c, n in aux_cameras_current
                                            ) + "]"
                                        logger.debug(
                                            f"  ★ [{self._ctrl} | cam {self._cam}] new best — tier={tier_idx} "
                                            f"LEDs{list(led_ids)} blobs[{b_anchor},{b1_ord},{b2_ord}] "
                                            f"sol={sol_i}  inliers={n_inlier_blobs}+{extra_inlier_count}aux  err={err:.3f}px  "
                                            f"cov={balanced_coverage:.2f} (led={led_cov:.2f} blob={blob_cov:.2f})"
                                            f"  matched={n_inlier_blobs}/{n_visible_leds}"
                                            + _aux_dbg
                                        )

                                    if best_error <= self._c_brute_strong_err and balanced_coverage >= self._c_brute_min_vis_cov:
                                        strong_found = True

                cur_prev_blob[triple_i] = blob_max
                if did_p3p:
                    tier_lq_tried[tier_idx] += 1
                if strong_found:
                    break

        # Joint LM refinement: optimise T_world_ctrl over all cameras simultaneously.
        if best_solution is not None:
            _joint_result = self._apply_joint_lm(
                best_solution['rvec'].reshape(3, 1),
                best_solution['tvec'].reshape(3),
                best_solution['assignment'],
                blobs,
                best_solution.get('aux_assignments') or {},
                other_cameras_blobs or [],
                K, dc, "Brute", best_solution['error'],
            )
            if _joint_result is not None:
                _rv_j, _tv_j, _err_j = _joint_result
                best_solution['rvec']   = _rv_j
                best_solution['tvec']   = _tv_j.reshape(3, 1)
                best_solution['error']  = _err_j
                best_solution['method'] = best_solution['method'] + '_mc'

        total_p3p_tried = sum(tier_p3p_calls)

        result_str = (
            f"found in tier_{solution_tier} ({_tier_label(self._c_brute_depth_tiers[solution_tier])})  "
            f"({best_inliers} inliers, {best_error:.2f} px)"
            if best_solution is not None else "not found"
        )
        dup_line = ""
        if bijection_counts is not None:
            n_unique = len(bijection_counts)
            n_dup    = total_p3p_tried - n_unique
            max_dup  = max(bijection_counts.values(), default=0)
            dup_line = (
                f"\n  bijections — {n_unique} unique / {total_p3p_tried} calls  "
                f"({n_dup} duplicate calls, max {max_dup}× same bijection)"
            )
        tier_lines = "\n".join(
            f"  tier_{i} ({_tier_label(self._c_brute_depth_tiers[i])}) — "
            f"{tier_p3p_calls[i]:>7} P3P calls  "
            f"({tier_lq_tried[i]}/{tier_lq_total[i]} LED triples reached inner loop)"
            for i in range(len(self._c_brute_depth_tiers))
        )
        logger.debug(
            f"[{self._ctrl} | cam {self._cam}] Brute-force: {result_str}\n"
            f"{tier_lines}\n"
            f"  total — {total_p3p_tried:>7} P3P calls"
            f"{dup_line}"
        )

        if best_solution is not None and blob_mask is not None:
            best_solution['_orig_idx'] = True
        return best_solution
