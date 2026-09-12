"""
pipeline.py — the end-to-end registration cascade.

  Stage 0  Metadata prior.       Both PDS labels carry the map-projected GSD, so
                                 the scale ratio is KNOWN, not something to
                                 discover. Pre-resampling the source to the
                                 reference GSD removes the single largest source
                                 of matcher failure before any feature is found.
                                 Skipping this is the most common design error.
  Stage 1  Illumination transform. Phase congruency (see photometric.py).
  Stage 2  Detect + describe.
  Stage 3  Match + MAGSAC++ on a progressive model ladder.
  Stage 4  Sub-pixel refinement by upsampled phase correlation.
  Stage 5  Uniformity enforcement, then refit on the uniform inlier set.
  Stage 6  Metrics.

Every stage is switchable so the ablation study can turn one thing off at a time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy import ndimage

from . import distribution as dist
from . import metrics as mt
from .features import detect_and_describe
from .matching import (estimate_progressive, estimate_model,
                       match_descriptors, refine_subpixel)
from .photometric import phase_congruency, gradient_flip_index


@dataclass
class PipelineConfig:
    method: str = 'rift'
    use_scale_prior: bool = True      # Stage 0
    subpixel: bool = True             # Stage 4
    uniform: bool = True              # Stage 5
    progressive_model: bool = True
    model: str = 'homography'
    ratio: float = 0.85
    ransac_thresh: float = 3.0
    n_features: int = 2500
    target_points: int = 400
    uniform_grid: int = 8
    subpixel_win: int = 32
    mim_shift_search: bool = True     # try cyclic MIM shifts (RIFT variants)
    d_azimuth_prior: float | None = None   # from PDS labels, if available


def _rift_match_best_shift(src_w, ref, cfg):
    """
    RIFT matching with a search over MIM orientation-bin shifts.

    When the solar azimuth difference is known from the PDS labels we test only
    the predicted shift and its two neighbours; otherwise all norient shifts.
    The winner is chosen by inlier count, so no metadata is strictly required.
    """
    from .features import pc_keypoints, mim_descriptor_bank, sun_bin_shift

    pcA, pcB = phase_congruency(src_w), phase_congruency(ref)
    norient = pcA['norient']
    if not cfg.mim_shift_search:
        shifts = [0]
    elif cfg.d_azimuth_prior is not None:
        p = sun_bin_shift(cfg.d_azimuth_prior, norient)
        shifts = sorted({(p - 1) % norient, p, (p + 1) % norient})
    else:
        shifts = list(range(norient))

    kA = pc_keypoints(pcA, n_max=cfg.n_features)
    kB = pc_keypoints(pcB, n_max=cfg.n_features)
    banksA, keptA = mim_descriptor_bank(pcA, kA, shifts)
    (descB,), keptB = mim_descriptor_bank(pcB, kB, [0])
    if len(keptA) < 8 or len(keptB) < 8:
        return None

    best = None
    for sh, descA in zip(shifts, banksA):
        pairs = match_descriptors(descA, descB, binary=False, ratio=cfg.ratio)
        if len(pairs) < 8:
            continue
        mA, mB = keptA[pairs[:, 0], :2], keptB[pairs[:, 1], :2]
        H, mask, model = estimate_progressive(mA, mB, cfg.ransac_thresh)
        if H is None:
            continue
        n_in = int(mask.sum())
        if best is None or n_in > best['n_inliers_raw']:
            best = dict(H=H, mask=mask, model=model, mA=mA, mB=mB,
                        n_inliers_raw=n_in, n_putative=len(pairs), mim_shift=int(sh))
    return best


def _prescale(src: np.ndarray, factor: float):
    """Resample source by `factor`, returning (image, H_pre) with H_pre exact."""
    if abs(factor - 1.0) < 1e-3:
        return src, np.eye(3)
    h, w = src.shape
    nh, nw = max(8, int(round(h * factor))), max(8, int(round(w * factor)))
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    out = cv2.resize(src, (nw, nh), interpolation=interp)
    # cv2.resize uses the pixel-area convention: x_out = fx*(x_in+0.5)-0.5
    fx, fy = nw / w, nh / h
    H = np.array([[fx, 0, 0.5 * fx - 0.5],
                  [0, fy, 0.5 * fy - 0.5],
                  [0, 0, 1.0]])
    return out, H


def register(src: np.ndarray, ref: np.ndarray, cfg: PipelineConfig | None = None,
             scale_prior: float | None = None, H_true: np.ndarray | None = None,
             verbose: bool = False) -> dict:
    """
    Register `src` onto `ref`.

    scale_prior : ref_gsd / src_gsd ratio from metadata, i.e. how much to shrink
                  the source so that one source pixel covers one reference pixel.
                  Pass None to disable and force the matcher to solve for scale.
    """
    cfg = cfg or PipelineConfig()
    t0 = time.time()
    out = dict(method=cfg.method, ok=False)

    # ---- Stage 0: metadata scale prior --------------------------------------
    if cfg.use_scale_prior and scale_prior is not None:
        src_w, H_pre = _prescale(src, 1.0 / scale_prior)
    else:
        src_w, H_pre = src, np.eye(3)
    out['src_shape_after_prior'] = src_w.shape

    # ---- dense / hybrid route ----------------------------------------------
    if cfg.method in ('dense', 'hybrid'):
        from .dense import coarse_align, dense_register
        t = time.time()
        if cfg.method == 'hybrid':
            # bootstrap the prior with RIFT, fall back to Fourier-Mellin
            b = _rift_match_best_shift(src_w, ref, cfg)
            H0 = b['H'] if b is not None else coarse_align(src_w, ref)
            out['bootstrap'] = 'rift' if b is not None else 'fourier-mellin'
        else:
            H0 = coarse_align(src_w, ref)
            out['bootstrap'] = 'fourier-mellin'
        out['t_coarse'] = time.time() - t
        if H_true is not None:
            out['rmse_coarse'] = mt.true_geometric_error(
                H0 @ H_pre, H_true, src.shape)['rmse_true']

        t = time.time()
        H_d, ps, pr, q = dense_register(src_w, ref, H0)
        out['t_dense'] = time.time() - t
        out['n_putative'] = int(len(ps))
        if len(ps) < 8:
            out['fail'] = 'dense matching failed'
            out['t_total'] = time.time() - t0
            return out

        out['uniformity_before'] = dist.uniformity_report(pr, ref.shape, cfg.uniform_grid)
        if cfg.uniform and len(ps) > cfg.target_points // 2:
            keep = dist.enforce_uniform(ps, pr, q, ref.shape,
                                        target=cfg.target_points, grid=cfg.uniform_grid)
            if len(keep) >= 12:
                ps, pr, q = ps[keep], pr[keep], q[keep]
        out['uniformity_after'] = dist.uniformity_report(pr, ref.shape, cfg.uniform_grid)

        H_fin, mask = estimate_model(ps, pr, 'homography', 1.5)
        if H_fin is None:
            H_fin, mask = H_d, np.ones(len(ps), bool)
        ps, pr = ps[mask], pr[mask]
        out['n_inliers_raw'] = int(mask.sum())
        out['inlier_ratio'] = float(mask.sum() / max(out['n_putative'], 1))
        out['model_used'] = 'homography'
        out['n_final'] = int(len(ps))
        out['ok'] = len(ps) >= 8
        H_final = H_fin @ H_pre
        out['H'] = H_final / H_final[2, 2]

        res = mt.residuals(H_fin, ps, pr)
        r_, lo, hi = mt.rmse_bootstrap_ci(res)
        out['rmse_reproj'] = r_
        out['rmse_ci'] = (lo, hi)
        out.update({f'resid_{k}': v for k, v in mt.residual_diagnostics(pr, res).items()})
        if H_true is not None:
            out.update(mt.true_geometric_error(out['H'], H_true, src.shape))
            origA = mt.apply_H(np.linalg.inv(H_pre), ps)
            out.update(mt.match_quality(origA, pr, H_true, tol=3.0))
        out['t_total'] = time.time() - t0
        return out

    # ---- Stage 1-3 ----------------------------------------------------------
    t = time.time()
    if cfg.method == 'rift':
        best = _rift_match_best_shift(src_w, ref, cfg)
        out['t_features'] = time.time() - t
        if best is None:
            out['fail'] = 'rift matching failed'
            out['t_total'] = time.time() - t0
            return out
        H_w, mask, model_used = best['H'], best['mask'], best['model']
        mA, mB = best['mA'], best['mB']
        out['n_putative'] = best['n_putative']
        out['mim_shift'] = best['mim_shift']
        out['n_kp_src'] = out['n_kp_ref'] = len(mA)
    else:
        ptsA, desA, binA, _ = detect_and_describe(src_w, cfg.method, cfg.n_features)
        ptsB, desB, binB, _ = detect_and_describe(ref, cfg.method, cfg.n_features)
        out['t_features'] = time.time() - t
        out['n_kp_src'], out['n_kp_ref'] = len(ptsA), len(ptsB)
        if len(ptsA) < 8 or len(ptsB) < 8:
            out['fail'] = 'too few keypoints'
            out['t_total'] = time.time() - t0
            return out

        pairs = match_descriptors(desA, desB, binary=binA, ratio=cfg.ratio)
        out['n_putative'] = len(pairs)
        if len(pairs) < 8:
            out['fail'] = 'too few putative matches'
            out['t_total'] = time.time() - t0
            return out
        mA, mB = ptsA[pairs[:, 0]], ptsB[pairs[:, 1]]
        if cfg.progressive_model:
            H_w, mask, model_used = estimate_progressive(mA, mB, cfg.ransac_thresh)
        else:
            H_w, mask = estimate_model(mA, mB, cfg.model, cfg.ransac_thresh)
            model_used = cfg.model
        if H_w is None:
            out['fail'] = 'model estimation failed'
            out['t_total'] = time.time() - t0
            return out

    out['model_used'] = model_used
    inA, inB = mA[mask], mB[mask]
    out['n_inliers_raw'] = int(mask.sum())
    out['inlier_ratio'] = float(mask.sum() / max(out['n_putative'], 1))

    # ---- Stage 4: sub-pixel refinement --------------------------------------
    if cfg.subpixel and len(inA) >= 8:
        t = time.time()
        refB, delta, qual, ok = refine_subpixel(
            src_w, ref, inA, inB, H_w, win=cfg.subpixel_win)
        out['t_subpixel'] = time.time() - t
        if ok.sum() >= 8:
            inA, inB = inA[ok], refB[ok]
            qual = qual[ok]
            out['n_subpixel_refined'] = int(ok.sum())
            out['mean_subpixel_shift'] = float(np.abs(delta[ok]).mean())
        else:
            qual = np.ones(len(inA))
            out['n_subpixel_refined'] = 0
    else:
        qual = np.ones(len(inA))
        out['n_subpixel_refined'] = 0

    # ---- Stage 5: uniformity ------------------------------------------------
    shape_ref = ref.shape
    out['uniformity_before'] = dist.uniformity_report(inB, shape_ref, cfg.uniform_grid)
    if cfg.uniform and len(inA) > cfg.target_points // 2:
        keep = dist.enforce_uniform(inA, inB, qual, shape_ref,
                                    target=cfg.target_points, grid=cfg.uniform_grid)
        if len(keep) >= 12:
            inA, inB = inA[keep], inB[keep]
    out['uniformity_after'] = dist.uniformity_report(inB, shape_ref, cfg.uniform_grid)

    # ---- final refit on the refined, uniform set ----------------------------
    H_fin, mask2 = estimate_model(inA, inB, model_used, max(1.5, cfg.ransac_thresh * 0.6))
    if H_fin is None:
        H_fin, mask2 = H_w, np.ones(len(inA), bool)
    inA, inB = inA[mask2], inB[mask2]

    # compose back through the Stage-0 prescale to get ORIGINAL-source geometry
    H_final = H_fin @ H_pre
    H_final = H_final / H_final[2, 2]
    out['H'] = H_final
    out['n_final'] = int(len(inA))
    out['ok'] = len(inA) >= 8

    # ---- Stage 6: metrics ---------------------------------------------------
    res = mt.residuals(H_fin, inA, inB)
    r, lo, hi = mt.rmse_bootstrap_ci(res)
    out['rmse_reproj'] = r
    out['rmse_ci'] = (lo, hi)
    out.update({f'resid_{k}': v for k, v in mt.residual_diagnostics(inB, res).items()})

    if H_true is not None:
        out.update(mt.true_geometric_error(H_final, H_true, src.shape))
        # correct-match statistics need points in ORIGINAL source coordinates
        Hpi = np.linalg.inv(H_pre)
        origA = mt.apply_H(Hpi, inA)
        out.update(mt.match_quality(origA, inB, H_true, tol=3.0))

    out['t_total'] = time.time() - t0
    if verbose:
        print({k: v for k, v in out.items() if k != 'H'})
    return out


def warp_source(src: np.ndarray, H: np.ndarray, ref_shape) -> np.ndarray:
    h, w = ref_shape
    return cv2.warpPerspective(src, H, (w, h), flags=cv2.INTER_CUBIC)


def checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 48) -> np.ndarray:
    """Standard visual QC mosaic: misalignment shows as broken edges at seams."""
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    m = (((yy // tile) + (xx // tile)) % 2).astype(bool)
    return np.where(m, a, b)
