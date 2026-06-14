import math

import cv2
import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment, least_squares
from scipy.spatial.distance import cdist
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from src.debug_config import is_deep, get_debug_triple, is_verbose_all, log_best
from src._pnp import _ransac_pnp, _project_points, _check_z_range
from src._visibility import _visible_mask, _cross_occluded_mask
from src._led_graph import _build_blob_neighbor_lists
from src.transformations import Transform


# ---------------------------------------------------------------------------
# brute_match gate helpers
# ---------------------------------------------------------------------------

def _gate_any_point(
    R_h: np.ndarray, tvec_h: np.ndarray,
    gate_obj: np.ndarray,
    gate_img: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    thresh_sq: float,
) -> Tuple[bool, float]:
    """
    Return True if ANY gate LED projects within sqrt(thresh_sq) pixels of ANY gate blob.
    If either pool is empty, returns True (no gate to fail).
    Returns (passed, min_dist_px); min_dist is only tracked in deep-debug mode.
    """
    if len(gate_obj) == 0 or len(gate_img) == 0:
        return True, 0.0

    track_dist = is_deep()
    min_dist   = np.inf if track_dist else 0.0
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
        _proj  = _project_points(_rv_ci, _tv_ci, _pts3d, _ocam.camera_matrix, _ocam.dist_coeffs)
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
        for (R_cw, t_cw, K_c, dc_c, leds, blobs_ref), w in zip(cam_data, cam_weights):
            R_ci = (R_cw @ R_wc).astype(np.float32)
            t_ci = (R_cw @ tv + t_cw).astype(np.float32)
            rv_ci = cv2.Rodrigues(R_ci)[0]
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


# ---------------------------------------------------------------------------
# proximity_match
# ---------------------------------------------------------------------------

def proximity_match(
    self,
    blobs: np.ndarray,
    predicted_pose: Tuple[np.ndarray, np.ndarray],
    blob_radii: Optional[np.ndarray] = None,
    blob_brightnesses: Optional[np.ndarray] = None,
    other_cameras_blobs: Optional[List] = None,
    occluders_per_cam: Optional[Dict[int, Tuple[np.ndarray, np.ndarray, object]]] = None,
    expansion_px: Optional[float] = None,
) -> Optional[Dict]:
    """
    Refine a predicted pose via global Hungarian matching over all model-visible LEDs.

    Projects every visible LED with the predicted pose, builds a cost matrix
    (distance + size + brightness penalties) against detected blobs, and solves
    globally via linear_sum_assignment. RANSAC filters inliers.
    Returns None when too few pairs survive; caller falls back to brute_match.
    """
    rvec_pred, tvec_pred = predicted_pose
    rvec_pred = np.asarray(rvec_pred, dtype=np.float32).reshape(3, 1)
    tvec_pred = np.asarray(tvec_pred, dtype=np.float32).reshape(3)

    K  = self.camera.camera_matrix
    dc = self.camera.dist_coeffs

    geom = self._geometry
    _cfg = getattr(self, '_matching_cfg', None) or {}
    _cam = self.camera.camera_idx
    _ctrl = self.model.name.replace("_controller", "")
    facing_threshold_deg         = float(_cfg.get('led_facing_angle_deg',              86.0))
    reprojection_threshold       = float(_cfg.get('proximity_reprojection_threshold',   2.0))
    min_inliers                  = int(  _cfg.get('proximity_min_inliers',
                                               _cfg.get('min_inliers', 4)))
    led_radius_mm                = float(_cfg.get('led_radius_mm',                      2.5))
    blob_size_max_factor         = float(_cfg.get('blob_size_max_factor',               4.0))
    blob_size_min_factor         = float(_cfg.get('blob_size_min_factor',               0.2))
    _log_size_filter             = is_deep() and bool(_cfg.get('log_size_filter', False))
    _br                          = float(_cfg.get('cross_occlusion_bounding_radius_m',  0.18))
    _gate_margin_px              = float(_cfg.get('cross_occlusion_gate_margin_px',    20.0))
    blob_size_score_weight       = float(_cfg.get('blob_size_score_weight',             0.5))
    blob_brightness_score_weight = float(_cfg.get('blob_brightness_score_weight',       0.3))
    proximity_expansion_px = expansion_px if expansion_px is not None \
                             else float(_cfg.get('proximity_expansion_px', 8.0))

    R_pred_arr, _ = cv2.Rodrigues(rvec_pred)
    focal_px = float(max(K[0, 0], K[1, 1]))

    vis_mask_pred = _visible_mask(
        R_pred_arr, tvec_pred,
        self.model.positions, self.model.normals,
        geom,
        cam_K=K, cam_dc=dc, cam_w=self.camera.width, cam_h=self.camera.height,
        cam_rpmax=self.camera.rpmax,
        facing_threshold_deg=facing_threshold_deg,
    )
    if occluders_per_cam:
        _occ = occluders_per_cam.get(self.camera.camera_idx)
        if _occ is not None:
            _R_occ, _t_occ, _geom_occ = _occ
            vis_mask_pred &= ~_cross_occluded_mask(
                R_pred_arr, tvec_pred, self.model.positions,
                _R_occ, _t_occ, _geom_occ,
                _br, _br, focal_px, _gate_margin_px,
                log_tag=f"[{_ctrl} | cam {_cam}]",
                vis_mask=vis_mask_pred,
            )
    n_model_visible = int(vis_mask_pred.sum())

    _brightness_norm = None
    if blob_brightnesses is not None and len(blob_brightnesses) > 0:
        _bmax = float(blob_brightnesses.max())
        if _bmax > 0:
            _brightness_norm = blob_brightnesses / _bmax

    # Hungarian: project all model-visible LEDs and match to detected blobs globally.
    pairs      = []
    locked_obj = []
    locked_img = []

    if n_model_visible > 0 and len(blobs) > 0:
        vis_ids     = np.where(vis_mask_pred)[0]
        vis_pos     = self.model.positions[vis_ids].astype(np.float32)
        proj_vis    = _project_points(rvec_pred, tvec_pred, vis_pos, K, dc)
        led_cam_vis = (R_pred_arr @ vis_pos.T).T + tvec_pred

        cost = cdist(proj_vis, blobs)

        if blob_radii is not None:
            for k in range(len(vis_ids)):
                depth       = float(max(led_cam_vis[k, 2], 0.01))
                expected_px = focal_px * (led_radius_mm / 1000.0) / depth
                ineligible  = (
                    (blob_radii < expected_px * blob_size_min_factor) |
                    (blob_radii > expected_px * blob_size_max_factor)
                )
                cost[k, ineligible] = 1e9
                if _log_size_filter:
                    for _c in np.where(ineligible)[0]:
                        logger.debug(
                            f"  LED {int(vis_ids[k])}: blob {_c} size-filtered"
                            f"  r={blob_radii[_c]:.2f}  expected"
                            f" {expected_px*blob_size_min_factor:.2f}–{expected_px*blob_size_max_factor:.2f}px"
                        )

        if _brightness_norm is not None:
            normals_cam = (R_pred_arr @ self.model.normals[vis_ids].T).T
            facing      = np.maximum(0.0, normals_cam[:, 2])
            cost += blob_brightness_score_weight * np.abs(facing[:, None] - _brightness_norm[None, :])

        n_vis    = len(vis_ids)
        cost_aug = np.hstack([cost, np.full((n_vis, n_vis), proximity_expansion_px - 1e-6)])
        row_ind, col_ind = linear_sum_assignment(cost_aug)
        for r, c in zip(row_ind, col_ind):
            if c < len(blobs) and cost[r, c] < proximity_expansion_px:
                led_id = int(vis_ids[r])
                pairs.append((c, led_id))
                locked_obj.append(self.model.positions[led_id])
                locked_img.append(blobs[c])

    logger.debug(
        f"[{_ctrl} | cam {_cam}] Proximity Hungarian: {len(pairs)}/{len(blobs)} blobs matched "
        f"of {n_model_visible} visible LEDs"
    )

    if len(pairs) < min_inliers:
        logger.debug(f"[{_ctrl} | cam {_cam}] Proximity: too few pairs ({len(pairs)} < {min_inliers}) → None")
        return None

    lo = np.array(locked_obj, dtype=np.float32)
    li = np.array(locked_img, dtype=np.float32)

    ok, rvec, tvec, ransac_idx = _ransac_pnp(
        lo, li, K, dc, rvec_pred, tvec_pred,
        reprojection_px=reprojection_threshold,
    )

    if not ok or ransac_idx is None:
        logger.debug(f"[{_ctrl} | cam {_cam}] Proximity: RANSAC failed → None")
        return None

    final_pairs = [pairs[k] for k in ransac_idx]
    if len(final_pairs) < min_inliers:
        logger.debug(f"[{_ctrl} | cam {_cam}] Proximity: RANSAC too few inliers ({len(final_pairs)}) → None")
        return None

    lo_f = self.model.positions[[l for _, l in final_pairs]].astype(np.float32)
    li_f = blobs[[b for b, _ in final_pairs]].astype(np.float32)
    pe   = np.linalg.norm(_project_points(rvec, tvec, lo_f, K, dc) - li_f, axis=1)
    error     = float(pe.mean())
    max_error = float(pe.max())

    # RANSAC-confirmed inlier LEDs used as anchors for aux camera snapping.
    final_lids = [lid for _, lid in final_pairs]
    final_obj  = self.model.positions[final_lids].astype(np.float32)

    aux_snapped_per_cam: Dict[int, List] = {}

    # Aux cameras: Hungarian over all model-visible LEDs × all aux blobs.
    # Same dummy-column pattern as the primary path prevents LEDs without a
    # nearby blob from stealing real blobs from LEDs that have one.
    if other_cameras_blobs:
        _R_ref, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32).reshape(3, 1))
        _T_world_ref = self.T_world_cam.compose(Transform(_R_ref, np.asarray(tvec, dtype=np.float32).reshape(3)))
        for _ocam, _oblobs, _oradii in other_cameras_blobs:
            _oblobs = np.asarray(_oblobs, dtype=np.float32)
            _pairs_i: List = []
            if len(_oblobs) > 0:
                _T_ci_r    = _ocam.T_world_cam.inverse().compose(_T_world_ref)
                _R_i_r     = _T_ci_r.R.astype(np.float32)
                _t_i_r     = _T_ci_r.t.astype(np.float32)
                _rv_i_r, _ = cv2.Rodrigues(_R_i_r)
                _focal_i_r = float(max(_ocam.camera_matrix[0, 0], _ocam.camera_matrix[1, 1]))

                _vis_i_r = _visible_mask(
                    _R_i_r, _t_i_r, self.model.positions, self.model.normals, geom,
                    cam_K=_ocam.camera_matrix, cam_dc=_ocam.dist_coeffs,
                    cam_w=_ocam.width, cam_h=_ocam.height, cam_rpmax=_ocam.rpmax,
                    facing_threshold_deg=facing_threshold_deg,
                )
                if occluders_per_cam:
                    _occ_i_r = occluders_per_cam.get(_ocam.camera_idx)
                    if _occ_i_r is not None:
                        _R_occ_i_r, _t_occ_i_r, _geom_occ_i_r = _occ_i_r
                        _vis_i_r &= ~_cross_occluded_mask(
                            _R_i_r, _t_i_r, self.model.positions,
                            _R_occ_i_r, _t_occ_i_r, _geom_occ_i_r,
                            _br, _br, _focal_i_r, _gate_margin_px,
                            log_tag=f"[{_ctrl} | cam {_cam} aux_cam {_ocam.camera_idx}]",
                            vis_mask=_vis_i_r,
                        )

                _vis_ids_i  = np.where(_vis_i_r)[0]
                if len(_vis_ids_i) > 0:
                    _vis_pos_i  = self.model.positions[_vis_ids_i].astype(np.float32)
                    _proj_i_r   = _project_points(_rv_i_r, _t_i_r, _vis_pos_i,
                                                  _ocam.camera_matrix, _ocam.dist_coeffs)
                    _led_ci_r   = (_R_i_r @ _vis_pos_i.T).T + _t_i_r
                    _exp_px_i_r = _focal_i_r * (led_radius_mm / 1000.0) / np.maximum(_led_ci_r[:, 2], 0.01)

                    _cost_i_r = cdist(_proj_i_r, _oblobs)

                    if _oradii is not None:
                        for _k in range(len(_vis_ids_i)):
                            _inelig_r = ((_oradii < _exp_px_i_r[_k] * blob_size_min_factor) |
                                         (_oradii > _exp_px_i_r[_k] * blob_size_max_factor))
                            _cost_i_r[_k, _inelig_r] = 1e9
                            if _log_size_filter:
                                for _c in np.where(_inelig_r)[0]:
                                    logger.debug(
                                        f"  LED {int(_vis_ids_i[_k])}: blob {_c}"
                                        f" size-filtered (aux cam{_ocam.camera_idx})"
                                        f"  r={_oradii[_c]:.2f}  expected"
                                        f" {_exp_px_i_r[_k]*blob_size_min_factor:.2f}–{_exp_px_i_r[_k]*blob_size_max_factor:.2f}px"
                                    )

                    _n_vis_i   = len(_vis_ids_i)
                    _cost_i_aug = np.hstack([_cost_i_r,
                                             np.full((_n_vis_i, _n_vis_i), proximity_expansion_px - 1e-6)])
                    _row_r, _col_r = linear_sum_assignment(_cost_i_aug)
                    for _rr, _cc in zip(_row_r, _col_r):
                        if _cc < len(_oblobs) and _cost_i_r[_rr, _cc] < proximity_expansion_px:
                            _pairs_i.append((int(_cc), int(_vis_ids_i[_rr])))

            aux_snapped_per_cam[_ocam.camera_idx] = _pairs_i
            if is_deep() and _pairs_i:
                logger.debug(
                    f"[{_ctrl} | cam {_cam}] Proximity aux snap cam{_ocam.camera_idx}: "
                    f"{len(_pairs_i)} pairs (refined pose)"
                    + ("  [size filter active]" if _oradii is not None else "")
                )

    aux_inlier_count = 0
    aux_cameras_result: List = []
    for _aux_cam_idx, _aux_pairs_i in aux_snapped_per_cam.items():
        _n_aux = len(_aux_pairs_i)
        aux_inlier_count += _n_aux
        aux_cameras_result.append((_aux_cam_idx, _n_aux))

    # Joint LM refinement: optimise T_world_ctrl over all cameras simultaneously.
    if bool(_cfg.get('joint_optimization', True)) and aux_snapped_per_cam:
        _aux_joint = [
            (_ocam, np.asarray(_oblobs, dtype=np.float32),
             aux_snapped_per_cam.get(_ocam.camera_idx, []))
            for _ocam, _oblobs, _ in (other_cameras_blobs or [])
            if aux_snapped_per_cam.get(_ocam.camera_idx)
        ]
        if _aux_joint:
            _R_pr, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32).reshape(3, 1))
            _T_wc_pr = self.T_world_cam.compose(
                Transform(_R_pr.astype(np.float64),
                          np.asarray(tvec, dtype=np.float64).reshape(3))
            )
            _aux_prefilter_px = float(_cfg.get('joint_aux_prefilter_px', 8.0))
            _aux_joint = _filter_aux_by_reprojection(
                _T_wc_pr, _aux_joint, self.model.positions, _aux_prefilter_px
            )
            _T_joint = _joint_err = None
            if _aux_joint:
                _T_joint, _joint_err = _joint_refine_pose(
                    _T_wc_pr, self.camera, final_pairs, blobs, _aux_joint, self.model.positions,
                    primary_weight=float(_cfg.get('joint_primary_weight', 2.0)),
                    huber_scale=float(_cfg.get('joint_huber_scale', 1.5)),
                )
            if _T_joint is not None:
                _T_prim = self.T_world_cam.inverse().compose(_T_joint)
                _rv_j = cv2.Rodrigues(_T_prim.R.astype(np.float32))[0]
                _tv_j = _T_prim.t.astype(np.float32)
                _proj_j = _project_points(_rv_j, _tv_j, lo_f, K, dc)
                _err_j = float(np.mean(np.linalg.norm(_proj_j - li_f, axis=1)))
                logger.debug(
                    f"[{_ctrl} | cam {_cam}] Proximity joint LM: "
                    f"primary err {error:.2f}→{_err_j:.2f}px  joint mean {_joint_err:.2f}px"
                )
                for _aux_cam_j, _aux_blobs_j, _aux_pr_j in _aux_joint:
                    if not _aux_pr_j:
                        continue
                    _T_aux_j = _aux_cam_j.T_world_cam.inverse().compose(_T_joint)
                    _rv_aux_j = cv2.Rodrigues(_T_aux_j.R.astype(np.float32))[0]
                    _tv_aux_j = _T_aux_j.t.astype(np.float32)
                    _leds_aux_j  = self.model.positions[[l for _, l in _aux_pr_j]].astype(np.float32)
                    _blobs_aux_j = _aux_blobs_j[[b for b, _ in _aux_pr_j]].astype(np.float32)
                    _proj_aux_j  = _project_points(_rv_aux_j, _tv_aux_j, _leds_aux_j,
                                                   _aux_cam_j.camera_matrix, _aux_cam_j.dist_coeffs)
                    _err_aux_j   = float(np.mean(np.linalg.norm(_proj_aux_j - _blobs_aux_j, axis=1)))
                    logger.debug(
                        f"[{_ctrl} | cam {_cam}] Proximity joint LM cam{_aux_cam_j.camera_idx}: "
                        f"err={_err_aux_j:.2f}px  ({len(_aux_pr_j)} pairs)"
                    )
                rvec, tvec, error = _rv_j, _tv_j, _err_j

    _aux_log = ""
    if aux_cameras_result:
        _aux_log = "  aux=[" + ",".join(f"cam{c}:{n}" for c, n in aux_cameras_result if n > 0) + "]"
    logger.debug(f"[{_ctrl} | cam {_cam}] Proximity: OK  inliers={len(final_pairs)}  err={error:.2f}px  max={max_error:.2f}px{_aux_log}")
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


# ---------------------------------------------------------------------------
# prior_constrained_match
# ---------------------------------------------------------------------------

def prior_constrained_match(
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
    _cam = self.camera.camera_idx
    _ctrl = self.model.name.replace("_controller", "")

    _cfg = getattr(self, '_matching_cfg', None) or {}
    reprojection_threshold = float(_cfg.get('reprojection_threshold', 2.0))
    led_radius_mm          = float(_cfg.get('led_radius_mm',          2.5))
    snap_factor            = float(_cfg.get('proximity_snap_factor',  4.0))
    blob_size_max_factor   = float(_cfg.get('blob_size_max_factor',   4.0))
    blob_size_min_factor   = float(_cfg.get('blob_size_min_factor',   0.2))
    blob_size_score_weight = float(_cfg.get('blob_size_score_weight', 0.5))

    mode       = 'p2p' if n_blobs >= 3 else 'p1p'
    n_required = 3     if mode == 'p2p' else 2

    # ── Primary snap ──────────────────────────────────────────────────────────
    prior_lids = [lid for _, lid in prior_assignment]
    prior_obj  = self.model.positions[prior_lids].astype(np.float32)
    proj_prior = _project_points(rvec_pred, tvec_pred, prior_obj, K, dc)
    led_cam    = (R_prior @ prior_obj.T).T + tvec_pred
    focal_px   = float(max(K[0, 0], K[1, 1]))

    locked_pairs: List[Tuple[int, int]] = []
    used_blobs: set = set()

    for i, lid in enumerate(prior_lids):
        if len(locked_pairs) >= n_required:
            break

        depth       = float(max(led_cam[i, 2], 0.01))
        expected_px = focal_px * (led_radius_mm / 1000.0) / depth
        snap_px     = expected_px * snap_factor

        if blob_radii is not None:
            eligible = (
                (blob_radii >= expected_px * blob_size_min_factor) &
                (blob_radii <= expected_px * blob_size_max_factor)
            )
        else:
            eligible = None

        dists = np.linalg.norm(blobs - proj_prior[i], axis=1)

        if eligible is not None and eligible.any():
            cand_idx   = np.where(eligible)[0]
            dists_cand = dists[cand_idx]
            size_err   = np.abs(blob_radii[cand_idx] - expected_px)
            scores     = dists_cand + blob_size_score_weight * size_err
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
                                           _ocam.camera_matrix, _ocam.dist_coeffs)
            _led_ci_p   = (_R_ci_p @ prior_obj.T).T + _t_ci_p
            _focal_ci   = float(max(_ocam.camera_matrix[0, 0], _ocam.camera_matrix[1, 1]))
            _pairs_ci: List = []
            _used_ci: set   = set()
            for _ii, _lid in enumerate(prior_lids):
                _depth_ci = float(max(_led_ci_p[_ii, 2], 0.01))
                _exp_ci   = _focal_ci * (led_radius_mm / 1000.0) / _depth_ci
                _snap_ci  = _exp_ci * snap_factor
                if _oradii is not None:
                    _elig_ci = ((_oradii >= _exp_ci * blob_size_min_factor) &
                                (_oradii <= _exp_ci * blob_size_max_factor))
                else:
                    _elig_ci = None
                _dists_ci = np.linalg.norm(_obl - _proj_ci_p[_ii], axis=1)
                if _elig_ci is not None and _elig_ci.any():
                    _cand_ci   = np.where(_elig_ci)[0]
                    _scores_ci = (_dists_ci[_cand_ci]
                                  + blob_size_score_weight * np.abs(_oradii[_cand_ci] - _exp_ci))
                    _jj        = int(_cand_ci[np.argmin(_scores_ci)])
                else:
                    _jj = int(np.argmin(_dists_ci))
                if _dists_ci[_jj] < _snap_ci and _jj not in _used_ci:
                    _pairs_ci.append((_jj, _lid))
                    _used_ci.add(_jj)
            _aux_pre[_ocam.camera_idx] = (_ocam, _obl, _pairs_ci)

    n_aux_pre = sum(len(v[2]) for v in _aux_pre.values())

    # ── Undistort primary hypothesis blob positions ───────────────────────────
    hyp_blobs  = np.array([blobs[b] for b, _ in pairs_hyp], dtype=np.float32)
    pts_undist = cv2.undistortPoints(
        hyp_blobs.reshape(-1, 1, 2), K, dc,
    ).reshape(-1, 2)

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
            _und_aux = cv2.undistortPoints(
                _aux_hyp.reshape(-1, 1, 2), _ocam.camera_matrix, _ocam.dist_coeffs,
            ).reshape(-1, 2)
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
    proj_all    = _project_points(rvec_pred, t_solved.astype(np.float32), all_obj, K, dc)
    errors      = np.linalg.norm(proj_all - all_img, axis=1)

    if np.any(errors > reprojection_threshold):
        logger.debug(
            f"prior_constrained ({mode}): validation failed "
            f"errors={errors.round(2)} thresh={reprojection_threshold} → None"
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
                                           _ocam.camera_matrix, _ocam.dist_coeffs)
            _led_ci_s   = (_R_ci_s @ prior_obj.T).T + _t_ci_s
            _focal_ci   = float(max(_ocam.camera_matrix[0, 0], _ocam.camera_matrix[1, 1]))
            _pairs_s: List = []
            _used_s: set   = set()
            for _ii, _lid in enumerate(prior_lids):
                _depth_s = float(max(_led_ci_s[_ii, 2], 0.01))
                _exp_s   = _focal_ci * (led_radius_mm / 1000.0) / _depth_s
                _snap_s  = _exp_s * snap_factor
                if _oradii is not None:
                    _elig_s = ((_oradii >= _exp_s * blob_size_min_factor) &
                               (_oradii <= _exp_s * blob_size_max_factor))
                else:
                    _elig_s = None
                _dists_s = np.linalg.norm(_obl - _proj_ci_s[_ii], axis=1)
                if _elig_s is not None and _elig_s.any():
                    _cand_s   = np.where(_elig_s)[0]
                    _scores_s = (_dists_s[_cand_s]
                                 + blob_size_score_weight * np.abs(_oradii[_cand_s] - _exp_s))
                    _jj_s     = int(_cand_s[np.argmin(_scores_s)])
                else:
                    _jj_s = int(np.argmin(_dists_s))
                if _dists_s[_jj_s] < _snap_s and _jj_s not in _used_s:
                    _pairs_s.append((_jj_s, _lid))
                    _used_s.add(_jj_s)
            aux_snapped_per_cam[_ocam.camera_idx] = _pairs_s

    aux_inlier_count   = sum(len(v) for v in aux_snapped_per_cam.values())
    aux_cameras_result = [(idx, len(pairs))
                          for idx, pairs in aux_snapped_per_cam.items() if pairs]

    _aux_log = ""
    if aux_cameras_result:
        _aux_log = "  aux=[" + ",".join(f"cam{c}:{n}" for c, n in aux_cameras_result) + "]"
    logger.debug(
        f"[{_ctrl} | cam {_cam}] prior_constrained ({mode}): OK  pairs={len(all_primary)}  err={error:.2f}px"
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


# ---------------------------------------------------------------------------
# brute_match
# ---------------------------------------------------------------------------

def brute_match(
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
    _cfg = getattr(self, '_matching_cfg', None) or {}
    _cam = self.camera.camera_idx
    _ctrl = self.model.name.replace("_controller", "")
    if _cfg:
        depth_tiers            = tuple(tuple(t) for t in _cfg.get('depth_tiers', depth_tiers))
        p4_threshold_px        = float(_cfg.get('p4_threshold_px',        p4_threshold_px))
        hungarian_threshold_px = float(_cfg.get('hungarian_threshold_px', hungarian_threshold_px))
        reprojection_threshold = float(_cfg.get('reprojection_threshold', reprojection_threshold))
        min_inliers            = int(  _cfg.get('min_inliers',            min_inliers))
        min_inlier_fraction    = _cfg.get('min_inlier_fraction', min_inlier_fraction) or None
        strong_match_inliers   = int(  _cfg.get('strong_match_inliers',   strong_match_inliers))
        strong_match_error_px  = float(_cfg.get('strong_match_error_px',  strong_match_error_px))
        min_vis_coverage       = float(_cfg.get('min_vis_coverage',       min_vis_coverage))
        rng_seed               = _cfg.get('rng_seed', rng_seed)
    facing_threshold_deg  = float(_cfg.get('led_facing_angle_deg', 86.0))
    blob_size_min_factor  = float(_cfg.get('blob_size_min_factor', 0.2))
    blob_size_max_factor  = float(_cfg.get('blob_size_max_factor', 4.0))
    _log_size_filter      = is_deep() and bool(_cfg.get('log_size_filter', False))
    _br                   = float(_cfg.get('cross_occlusion_bounding_radius_m', 0.18))
    _gate_margin_px       = float(_cfg.get('cross_occlusion_gate_margin_px',   20.0))
    led_radius_mm         = float(_cfg.get('led_radius_mm',        2.5))
    # Aux-camera inlier threshold for step 6.7: looser than primary RANSAC threshold.
    brute_aux_reproj_px   = float(_cfg.get('joint_aux_prefilter_px', reprojection_threshold * 2.0))

    blobs   = np.asarray(blobs, dtype=np.float32)
    n_blobs = len(blobs)
    _avail_idx  = np.where(blob_mask)[0].astype(np.int32) if blob_mask is not None else np.arange(n_blobs, dtype=np.int32)
    n_available = len(_avail_idx)
    if n_available < 4:
        return None

    if min_inlier_fraction is not None:
        fraction_floor     = int(np.ceil(min_inlier_fraction * n_available))
        min_inliers_eff    = max(min_inliers, fraction_floor)
        strong_inliers_eff = min(strong_match_inliers, fraction_floor)
    else:
        min_inliers_eff    = min_inliers
        strong_inliers_eff = strong_match_inliers

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

    max_blob_depth = max(t[1] for t in depth_tiers)
    blob_nbr = _build_blob_neighbor_lists(blobs, k=max_blob_depth)

    # Undistort blobs once (mirrors OpenHMD's correspondence_search_set_blobs).
    # Gate check uses pinhole projection which is valid in undistorted space.
    blobs_undist = cv2.undistortPoints(
        blobs.reshape(-1, 1, 2), K, dc, P=K
    ).reshape(-1, 2).astype(np.float32)

    p4_thresh_sq = p4_threshold_px ** 2
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

    rng = np.random.default_rng(rng_seed)

    tier_p3p_calls = [0] * len(depth_tiers)
    tier_lq_tried  = [0] * len(depth_tiers)
    tier_lq_total  = [0] * len(depth_tiers)

    _dbg_leds, _dbg_blobs = get_debug_triple()
    debug_active      = _dbg_leds is not None or _dbg_blobs is not None
    debug_led_anchor  = int(_dbg_leds[0])     if _dbg_leds  is not None else None
    debug_led_set     = frozenset(_dbg_leds)  if _dbg_leds  is not None else None
    debug_blob_anchor = int(_dbg_blobs[0])    if _dbg_blobs is not None else None
    debug_blob_set    = frozenset(_dbg_blobs) if _dbg_blobs is not None else None

    seen_bijections:  set                  = set()
    bijection_counts: Dict[frozenset, int] = {} if is_deep() else None

    for tier_idx, tier_spec in enumerate(depth_tiers):
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
                        n_sols, rvecs, tvecs = cv2.solveP3P(
                            p3p_world_pts.reshape(3, 1, 3),
                            p3p_img_pts.reshape(3, 1, 2),
                            K, dc,
                            flags=cv2.SOLVEPNP_P3P,
                        )
                        if dbg:
                            logger.debug(f"  P3P returned {n_sols} solutions")
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
                            if dbg:
                                logger.debug(f"  sol {sol_i}: z={tvec_h[2]:.3f} m  depth_ok={z_ok}")
                            if not z_ok:
                                continue

                            R_h, _ = cv2.Rodrigues(rvec_h)

                            # ── 3. Gate check (any gate LED near any gate blob) ─
                            gate_ok, gate_dist = _gate_any_point(R_h, tvec_h, gate_led_world_pts, gate_blob_img_pts, fx, fy, cx, cy, p4_thresh_sq)
                            if dbg:
                                logger.debug(f"  sol {sol_i}: gate_ok={gate_ok}, dist={gate_dist:.2f}px")
                            if not gate_ok:
                                continue

                            # ── 4. Full inlier count on all visible LEDs ───────
                            vis_mask_h = _visible_mask(
                                R_h, tvec_h, positions, normals,
                                geom,
                                cam_K=K, cam_dc=dc, cam_w=self.camera.width, cam_h=self.camera.height,
                                cam_rpmax=self.camera.rpmax,
                                facing_threshold_deg=facing_threshold_deg,
                            )
                            vis_ids = np.where(vis_mask_h)[0]
                            if dbg:
                                logger.debug(f"  sol {sol_i}: {len(vis_ids)} visible LEDs")
                            if len(vis_ids) < min_inliers:
                                continue

                            proj_all = _project_points(rvec_h, tvec_h, positions[vis_ids], K, dc)
                            cost     = cdist(blobs, proj_all)
                            if blob_mask is not None:
                                _cost_sub = cost[_avail_idx]
                                _sub_rows, hungarian_led_cols = linear_sum_assignment(_cost_sub)
                                hungarian_blob_rows = _avail_idx[_sub_rows]
                            else:
                                hungarian_blob_rows, hungarian_led_cols = linear_sum_assignment(cost)

                            inlier_mask    = cost[hungarian_blob_rows, hungarian_led_cols] < hungarian_threshold_px
                            inlier_blobs   = hungarian_blob_rows[inlier_mask]
                            inlier_leds    = vis_ids[hungarian_led_cols[inlier_mask]]

                            if dbg:
                                outlier_mask     = cost[hungarian_blob_rows, hungarian_led_cols] >= hungarian_threshold_px
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
                                reprojection_px=reprojection_threshold,
                            )
                            if dbg:
                                logger.debug(f"  sol {sol_i}: RANSAC ok={ok_r}, "
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
                            vis_mask_r = _visible_mask(
                                R_r, tvec_r_flat, positions, normals,
                                geom,
                                cam_K=K, cam_dc=dc, cam_w=self.camera.width, cam_h=self.camera.height,
                                cam_rpmax=self.camera.rpmax,
                                facing_threshold_deg=facing_threshold_deg,
                            )
                            if occluders_per_cam:
                                _occ_r = occluders_per_cam.get(self.camera.camera_idx)
                                if _occ_r is not None:
                                    _R_occ_r, _t_occ_r, _geom_occ_r = _occ_r
                                    vis_mask_r &= ~_cross_occluded_mask(
                                        R_r, tvec_r_flat, positions,
                                        _R_occ_r, _t_occ_r, _geom_occ_r,
                                        _br, _br, focal_px, _gate_margin_px,
                                        log_tag=f"[{_ctrl} | cam {_cam}]",
                                        vis_mask=vis_mask_r,
                                    )
                            # Drop inliers that became occluded under the refined pose.
                            inlier_still_visible = vis_mask_r[inlier_leds]
                            inlier_leds  = inlier_leds[inlier_still_visible]
                            inlier_blobs = inlier_blobs[inlier_still_visible]
                            if dbg:
                                logger.debug(f"  sol {sol_i}: {len(inlier_blobs)} inliers after vis recheck")
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
                                proj_vis_r = _project_points(rvec_r, tvec_r, positions[vis_ids_r], K, dc)
                                cost_r     = cdist(blobs, proj_vis_r)
                                sub_min    = cost_r[np.ix_(unmatched_blobs, unmatched_col_idx)].min(axis=0)
                                _led_cam_r = ((R_r @ positions[vis_ids_r].T).T + tvec_r_flat
                                              if blob_radii is not None else None)
                                extra_blobs: List[int] = []
                                extra_leds:  List[int] = []
                                for order_j in np.argsort(sub_min):
                                    if sub_min[order_j] >= reprojection_threshold:
                                        break
                                    col    = int(unmatched_col_idx[order_j])
                                    led_id = int(vis_ids_r[col])
                                    _exp_px_rc = (focal_px * (led_radius_mm / 1000.0) /
                                                  float(max(_led_cam_r[col, 2], 0.01))
                                                  if _led_cam_r is not None else None)
                                    for row_i in np.argsort(cost_r[unmatched_blobs, col]):
                                        b = int(unmatched_blobs[row_i])
                                        if b in matched_blob_set:
                                            continue
                                        if cost_r[b, col] >= reprojection_threshold:
                                            break
                                        if (_exp_px_rc is not None and not (
                                                _exp_px_rc * blob_size_min_factor
                                                <= float(blob_radii[b])
                                                <= _exp_px_rc * blob_size_max_factor)):
                                            if _log_size_filter:
                                                logger.debug(
                                                    f"  LED {led_id}: blob {b}"
                                                    f" size-filtered (brute extra-blob)"
                                                    f"  r={float(blob_radii[b]):.2f}  expected"
                                                    f" {_exp_px_rc*blob_size_min_factor:.2f}–{_exp_px_rc*blob_size_max_factor:.2f}px"
                                                )
                                            continue
                                        matched_blob_set.add(b)
                                        extra_blobs.append(b)
                                        extra_leds.append(led_id)
                                        break
                                if extra_blobs:
                                    inlier_blobs = np.concatenate([inlier_blobs, np.array(extra_blobs, dtype=inlier_blobs.dtype)])
                                    inlier_leds  = np.concatenate([inlier_leds,  np.array(extra_leds,  dtype=inlier_leds.dtype)])
                                    if dbg:
                                        logger.debug(f"  sol {sol_i}: +{len(extra_blobs)} blob(s) recovered post-RANSAC")

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
                                    _T_ci_ctrl = _ocam.T_world_cam.inverse().compose(T_world_ctrl)
                                    _R_i  = _T_ci_ctrl.R.astype(np.float32)
                                    _t_i  = _T_ci_ctrl.t.astype(np.float32)
                                    _rv_i = cv2.Rodrigues(_R_i)[0]

                                    _vis_i = _visible_mask(
                                        _R_i, _t_i, positions, normals, geom,
                                        cam_K=_ocam.camera_matrix,
                                        cam_dc=_ocam.dist_coeffs,
                                        cam_w=_ocam.width,
                                        cam_h=_ocam.height,
                                        cam_rpmax=_ocam.rpmax,
                                        facing_threshold_deg=facing_threshold_deg,
                                    )
                                    if occluders_per_cam:
                                        _occ_i = occluders_per_cam.get(_ocam.camera_idx)
                                        if _occ_i is not None:
                                            _R_occ_i, _t_occ_i, _geom_occ_i = _occ_i
                                            _focal_i_b = float(max(_ocam.camera_matrix[0, 0],
                                                                    _ocam.camera_matrix[1, 1]))
                                            _vis_i &= ~_cross_occluded_mask(
                                                _R_i, _t_i, positions,
                                                _R_occ_i, _t_occ_i, _geom_occ_i,
                                                _br, _br, _focal_i_b, _gate_margin_px,
                                                log_tag=f"[{_ctrl} | cam {_cam} aux_cam {_ocam.camera_idx}]",
                                                vis_mask=_vis_i,
                                            )
                                    _vis_ids_i = np.where(_vis_i)[0]
                                    if len(_vis_ids_i) == 0:
                                        continue

                                    _proj_i = _project_points(
                                        _rv_i, _t_i, positions[_vis_ids_i],
                                        _ocam.camera_matrix, _ocam.dist_coeffs,
                                    )
                                    _cost_i = cdist(_oblobs, _proj_i)
                                    _rows_i, _cols_i = linear_sum_assignment(_cost_i)
                                    _inlier_i = _cost_i[_rows_i, _cols_i] < brute_aux_reproj_px
                                    _led_cam_i = (_R_i @ positions[_vis_ids_i].T).T + _t_i
                                    if _oradii is not None:
                                        _focal_i = float(max(_ocam.camera_matrix[0, 0], _ocam.camera_matrix[1, 1]))
                                        for _k in range(len(_rows_i)):
                                            if not _inlier_i[_k]:
                                                continue
                                            _depth_k = float(max(_led_cam_i[_cols_i[_k], 2], 0.01))
                                            _exp_px_k = _focal_i * (led_radius_mm / 1000.0) / _depth_k
                                            if not (_exp_px_k * blob_size_min_factor
                                                    <= float(_oradii[_rows_i[_k]])
                                                    <= _exp_px_k * blob_size_max_factor):
                                                if _log_size_filter:
                                                    logger.debug(
                                                        f"  LED {int(_vis_ids_i[_cols_i[_k]])}: blob {int(_rows_i[_k])}"
                                                        f" size-filtered (brute aux-inlier cam{_ocam.camera_idx})"
                                                        f"  r={float(_oradii[_rows_i[_k]]):.2f}  expected"
                                                        f" {_exp_px_k*blob_size_min_factor:.2f}–{_exp_px_k*blob_size_max_factor:.2f}px"
                                                    )
                                                _inlier_i[_k] = False
                                    _n_aux = int(_inlier_i.sum())
                                    if dbg:
                                        _matched_dists = _cost_i[_rows_i, _cols_i]
                                        logger.debug(
                                            f"  sol {sol_i}: aux cam{_ocam.camera_idx} "
                                            f"vis={len(_vis_ids_i)} blobs={len(_oblobs)} "
                                            f"matched_dists={_matched_dists.round(1).tolist()} "
                                            f"thresh={brute_aux_reproj_px:.1f}px → {_n_aux} inliers"
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
                            if balanced_coverage < min_vis_coverage:
                                if dbg:
                                    logger.debug(
                                        f"  sol {sol_i}: balanced coverage {balanced_coverage:.2f} < {min_vis_coverage:.2f}"
                                        f"  led_cov={led_cov:.2f}  blob_cov={blob_cov:.2f}"
                                        f"  (primary {n_inlier_blobs}/{n_visible_leds} leds,"
                                        f" {n_inlier_blobs}/{min(n_available, n_visible_leds)} blobs"
                                        f" +{extra_inlier_count} aux)"
                                    )
                                continue

                            proj_r = _project_points(rvec_r, tvec_r, positions[inlier_leds], K, dc)
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

                            if dbg:
                                logger.debug(f"  sol {sol_i}: err={err:.3f} px  "
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
                                        f"  ★ [{_ctrl} | cam {_cam}] new best — tier={tier_idx} "
                                        f"LEDs{list(led_ids)} blobs[{b_anchor},{b1_ord},{b2_ord}] "
                                        f"sol={sol_i}  inliers={n_inlier_blobs}+{extra_inlier_count}aux  err={err:.3f}px  "
                                        f"cov={balanced_coverage:.2f} (led={led_cov:.2f} blob={blob_cov:.2f})"
                                        f"  matched={n_inlier_blobs}/{n_visible_leds}"
                                        + _aux_dbg
                                    )

                                if best_error <= strong_match_error_px and balanced_coverage >= min_vis_coverage:
                                    strong_found = True

            cur_prev_blob[triple_i] = blob_max
            if did_p3p:
                tier_lq_tried[tier_idx] += 1
            if strong_found:
                break

    # Joint LM refinement: optimise T_world_ctrl over all cameras simultaneously.
    if (best_solution is not None
            and other_cameras_blobs
            and bool(_cfg.get('joint_optimization', True))
            and best_solution.get('aux_assignments')):
        _aux_joint = [
            (_ocam, np.asarray(_oblobs, dtype=np.float32),
             best_solution['aux_assignments'].get(_ocam.camera_idx, []))
            for _ocam, _oblobs, _ in other_cameras_blobs
            if best_solution['aux_assignments'].get(_ocam.camera_idx)
        ]
        if _aux_joint:
            _R_b, _ = cv2.Rodrigues(best_solution['rvec'].reshape(3, 1).astype(np.float32))
            _T_wc_b = self.T_world_cam.compose(
                Transform(_R_b.astype(np.float64),
                          best_solution['tvec'].reshape(3).astype(np.float64))
            )
            _aux_prefilter_px = float(_cfg.get('joint_aux_prefilter_px', 8.0))
            _aux_joint = _filter_aux_by_reprojection(
                _T_wc_b, _aux_joint, positions, _aux_prefilter_px
            )
            _T_joint, _joint_err = _joint_refine_pose(
                _T_wc_b,
                self.camera,
                best_solution['assignment'],
                blobs,
                _aux_joint,
                positions,
                primary_weight=float(_cfg.get('joint_primary_weight', 2.0)),
                huber_scale=float(_cfg.get('joint_huber_scale', 1.5)),
            ) if _aux_joint else (None, None)
            if _T_joint is not None:
                _T_prim = self.T_world_cam.inverse().compose(_T_joint)
                _rv_j = cv2.Rodrigues(_T_prim.R.astype(np.float32))[0]
                _tv_j = _T_prim.t.astype(np.float32)
                _lo_b = positions[np.array([l for _, l in best_solution['assignment']], dtype=np.int32)]
                _li_b = blobs[np.array([b for b, _ in best_solution['assignment']], dtype=np.int32)]
                _err_j = float(np.mean(np.linalg.norm(
                    _project_points(_rv_j, _tv_j, _lo_b, K, dc) - _li_b, axis=1,
                )))
                logger.debug(
                    f"[{_ctrl} | cam {_cam}] Brute joint LM: "
                    f"primary err {best_solution['error']:.2f}→{_err_j:.2f}px  joint mean {_joint_err:.2f}px"
                )
                for _aux_cam_j, _aux_blobs_j, _aux_pr_j in _aux_joint:
                    if not _aux_pr_j:
                        continue
                    _T_aux_j = _aux_cam_j.T_world_cam.inverse().compose(_T_joint)
                    _rv_aux_j = cv2.Rodrigues(_T_aux_j.R.astype(np.float32))[0]
                    _tv_aux_j = _T_aux_j.t.astype(np.float32)
                    _leds_aux_j  = positions[[l for _, l in _aux_pr_j]].astype(np.float32)
                    _blobs_aux_j_ref = _aux_blobs_j[[b for b, _ in _aux_pr_j]].astype(np.float32)
                    _proj_aux_j  = _project_points(_rv_aux_j, _tv_aux_j, _leds_aux_j,
                                                   _aux_cam_j.camera_matrix, _aux_cam_j.dist_coeffs)
                    _err_aux_j   = float(np.mean(np.linalg.norm(_proj_aux_j - _blobs_aux_j_ref, axis=1)))
                    logger.debug(
                        f"[{_ctrl} | cam {_cam}] Brute joint LM cam{_aux_cam_j.camera_idx}: "
                        f"err={_err_aux_j:.2f}px  ({len(_aux_pr_j)} pairs)"
                    )
                best_solution['rvec']   = _rv_j
                best_solution['tvec']   = _tv_j.reshape(3, 1)
                best_solution['error']  = _err_j
                best_solution['method'] = best_solution['method'] + '_mc'

    total_p3p_tried = sum(tier_p3p_calls)

    result_str = (
        f"found in tier_{solution_tier} ({_tier_label(depth_tiers[solution_tier])})  "
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
        f"  tier_{i} ({_tier_label(depth_tiers[i])}) — "
        f"{tier_p3p_calls[i]:>7} P3P calls  "
        f"({tier_lq_tried[i]}/{tier_lq_total[i]} LED triples reached inner loop)"
        for i in range(len(depth_tiers))
    )
    logger.debug(
        f"[{_ctrl} | cam {_cam}] Brute-force: {result_str}\n"
        f"{tier_lines}\n"
        f"  total — {total_p3p_tried:>7} P3P calls"
        f"{dup_line}"
    )

    # if best_solution is not None:
    #     R_best, _ = cv2.Rodrigues(best_solution["rvec"])
    #     tvec_best = best_solution["tvec"].reshape(3)
    #     cam_dbg = -(R_best.T @ tvec_best)
    #     print(f"[vis check] cam_world = {cam_dbg.round(4)}")
    #     print("[frustum debug] inner LED occlusion check for best solution:")
    #     _visible_mask(R_best, tvec_best, positions, normals, geom, debug=True)

    if best_solution is not None and blob_mask is not None:
        best_solution['_orig_idx'] = True
    return best_solution
