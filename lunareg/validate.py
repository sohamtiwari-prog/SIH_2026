"""
validate.py — checking a registration when there is no truth to check against.

The problem, restated from the measurements
-------------------------------------------
A registration can be 250-760 px wrong while reporting a healthy inlier ratio,
a tight bootstrap interval and hundreds of tie points. Every diagnostic used so
far is a measure of SELF-CONSISTENCY: it asks whether the surviving matches
agree with each other. They can agree perfectly and all be wrong together.

What is needed is something that can disagree with itself.

Cycle consistency
-----------------
Register source onto reference, then register reference onto source. Compose the
two transforms. If both are correct the composition is the identity, because you
have gone somewhere and come back. Any deviation is error the pipeline generated
without being told the answer.

The critical property is that the two runs are not the same computation. The
prior differs, the template windows are cut from different images, the phase
congruency is computed on different data, and the RANSAC draws differ. Two
independent estimates that agree are evidence; one estimate agreeing with itself
is not.

Split-half
----------
Fit the geometric model on a random half of the tie points and measure
reprojection error on the held-out half. This catches overfitting — a homography
bending to accommodate noise looks excellent on the points that shaped it.
"""

from __future__ import annotations

import numpy as np

from .metrics import apply_H
from .pipeline import PipelineConfig, register


def composition_error(H_fwd: np.ndarray, H_bwd: np.ndarray, shape,
                      n: int = 49) -> dict:
    """
    Deviation of H_bwd o H_fwd from the identity, sampled over the source frame.

    Returned in SOURCE pixels, which is the frame the user cares about.
    """
    if H_fwd is None or H_bwd is None:
        return dict(cycle_rmse=None, cycle_max=None)
    h, w = shape
    g = int(np.sqrt(n))
    xs = np.linspace(0.05 * w, 0.95 * w, g)
    ys = np.linspace(0.05 * h, 0.95 * h, g)
    gx, gy = np.meshgrid(xs, ys)
    P = np.stack([gx.ravel(), gy.ravel()], 1)
    try:
        Q = apply_H(H_bwd @ H_fwd, P)
    except Exception:
        return dict(cycle_rmse=None, cycle_max=None)
    d = np.linalg.norm(Q - P, axis=1)
    if not np.isfinite(d).all():
        return dict(cycle_rmse=None, cycle_max=None)
    return dict(cycle_rmse=float(np.sqrt((d ** 2).mean())),
                cycle_max=float(d.max()))


def split_half_error(ptsA: np.ndarray, ptsB: np.ndarray, model: str = 'homography',
                     n_rep: int = 12, seed: int = 0) -> float | None:
    """Median held-out reprojection RMSE over repeated random halves."""
    from .matching import estimate_model
    n = len(ptsA)
    if n < 20:
        return None
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_rep):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2:]
        H, _ = estimate_model(ptsA[a], ptsB[a], model, 2.0)
        if H is None:
            continue
        d = apply_H(H, ptsA[b]) - ptsB[b]
        errs.append(float(np.sqrt((d ** 2).sum(1).mean())))
    return float(np.median(errs)) if errs else None


def cycle_check(src: np.ndarray, ref: np.ndarray, cfg: PipelineConfig | None = None,
                scale_prior: float | None = None, H_true: np.ndarray | None = None) -> dict:
    """
    Run the pipeline both ways and report every truth-free consistency signal
    alongside the true error, so the two can be correlated afterwards.
    """
    cfg = cfg or PipelineConfig(method='dense')
    fwd = register(src, ref, cfg, scale_prior=scale_prior, H_true=H_true)
    inv_prior = (1.0 / scale_prior) if scale_prior else None
    bwd = register(ref, src, cfg, scale_prior=inv_prior)

    out = dict(fwd_ok=bool(fwd.get('ok')), bwd_ok=bool(bwd.get('ok')))
    for k in ('rmse_true', 'max_true', 'rmse_reproj', 'inlier_ratio',
              'n_final', 'n_putative'):
        out[k] = fwd.get(k)
    out['rmse_reproj_bwd'] = bwd.get('rmse_reproj')
    out['n_final_bwd'] = bwd.get('n_final')

    out.update(composition_error(fwd.get('H'), bwd.get('H'), src.shape))

    # scale agreement: the two directions should report reciprocal scales
    Hf, Hb = fwd.get('H'), bwd.get('H')
    if Hf is not None and Hb is not None:
        sf = float(np.hypot(Hf[0, 0], Hf[1, 0]))
        sb = float(np.hypot(Hb[0, 0], Hb[1, 0]))
        out['scale_product'] = sf * sb
        out['scale_log_dev'] = float(abs(np.log(max(sf * sb, 1e-9))))
    else:
        out['scale_product'] = None
        out['scale_log_dev'] = None
    return out
