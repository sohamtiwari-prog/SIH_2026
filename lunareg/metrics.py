"""
metrics.py — evaluation.

Two families:

  Ground-truth metrics (synthetic harness only)
    True RMSE against the known homography. This is the honest number.

  Self-consistency metrics (available on real data)
    Reprojection RMSE against the FITTED model, inlier ratio, NCM, CMR.
    These are what you can report on real OHRC/NAC pairs — but note the trap:
    reprojection RMSE is computed on the inliers of the model it is measuring,
    so it is optimistically biased. We therefore also report leave-one-out and
    checkerboard-consistency numbers, which are not self-referential.

Residual diagnostics answer a different question than "how big is the error":
they ask "is the error structured?". Structured residuals mean the model is
wrong (an affine fit to a scene with relief, say), not that the points are
noisy. Moran's I on the residual field detects exactly that.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def apply_H(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    P = np.hstack([pts, np.ones((len(pts), 1))]).T
    Q = H @ P
    return (Q[:2] / Q[2]).T


def residuals(H: np.ndarray, ptsA: np.ndarray, ptsB: np.ndarray) -> np.ndarray:
    return apply_H(H, ptsA) - ptsB


def rmse(res: np.ndarray) -> float:
    if len(res) == 0:
        return float('nan')
    return float(np.sqrt((res ** 2).sum(axis=1).mean()))


def true_geometric_error(H_est, H_true, shape, n: int = 64) -> dict:
    """
    Corner/grid transfer error: sample a grid in the source frame, push through
    both homographies, and measure the displacement. Independent of which points
    were matched, so it cannot be gamed by a lucky inlier set.
    """
    if H_est is None:
        return dict(rmse_true=float('nan'), max_true=float('nan'),
                    bias_x=float('nan'), bias_y=float('nan'))
    h, w = shape
    xs = np.linspace(0.05 * w, 0.95 * w, int(np.sqrt(n)))
    ys = np.linspace(0.05 * h, 0.95 * h, int(np.sqrt(n)))
    gx, gy = np.meshgrid(xs, ys)
    P = np.stack([gx.ravel(), gy.ravel()], 1)
    d = apply_H(H_est, P) - apply_H(H_true, P)
    return dict(rmse_true=float(np.sqrt((d ** 2).sum(1).mean())),
                max_true=float(np.sqrt((d ** 2).sum(1)).max()),
                bias_x=float(d[:, 0].mean()), bias_y=float(d[:, 1].mean()))


def match_quality(ptsA, ptsB, H_true, tol: float = 3.0) -> dict:
    """
    Correct-match statistics using ground truth.
      NCM  number of correct matches
      CMR  correct match ratio (precision of the putative set)
    """
    if len(ptsA) == 0:
        return dict(ncm=0, cmr=0.0, n_put=0)
    d = np.linalg.norm(apply_H(H_true, ptsA) - ptsB, axis=1)
    ncm = int((d <= tol).sum())
    return dict(ncm=ncm, cmr=float(ncm / len(ptsA)), n_put=int(len(ptsA)))


# --------------------------------------------------------------------------
# Residual structure
# --------------------------------------------------------------------------

def morans_I(pts: np.ndarray, values: np.ndarray, k: int = 8) -> float:
    """
    Moran's I of a scalar residual field over a k-NN spatial weight graph.

    I ~ 0    residuals are spatially random -> model captures the geometry.
    I >> 0   residuals cluster -> systematic, unmodelled distortion remains.
    """
    n = len(pts)
    if n < k + 2:
        return float('nan')
    z = values - values.mean()
    denom = (z ** 2).sum()
    if denom < 1e-12:
        return float('nan')
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)
    idx = idx[:, 1:]
    num = float(sum(z[i] * z[idx[i]].sum() for i in range(n)))
    W = n * k
    return float((n / W) * (num / denom))


def residual_diagnostics(pts: np.ndarray, res: np.ndarray) -> dict:
    if len(res) < 10:
        return dict(moran_x=float('nan'), moran_y=float('nan'),
                    bias=float('nan'), aniso=float('nan'))
    bias = float(np.linalg.norm(res.mean(axis=0)))
    C = np.cov(res.T)
    ev = np.linalg.eigvalsh(C)
    aniso = float(np.sqrt(max(ev[-1], 1e-12) / max(ev[0], 1e-12)))
    return dict(moran_x=morans_I(pts, res[:, 0]),
                moran_y=morans_I(pts, res[:, 1]),
                bias=bias, aniso=aniso)


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI. Reported because single-run RMSE means little."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return (float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n_boot, len(v)))
    bs = stat(v[idx], axis=1)
    return (float(stat(v)), float(np.percentile(bs, 100 * alpha / 2)),
            float(np.percentile(bs, 100 * (1 - alpha / 2))))


def rmse_bootstrap_ci(res: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """CI on RMSE itself, resampling correspondences."""
    if len(res) < 4:
        return (float('nan'),) * 3
    sq = (res ** 2).sum(axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sq), (n_boot, len(sq)))
    bs = np.sqrt(sq[idx].mean(axis=1))
    return (float(np.sqrt(sq.mean())), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


def loo_rmse(ptsA, ptsB, fit_fn, max_pts: int = 120, seed: int = 0) -> float:
    """
    Leave-one-out reprojection RMSE — the non-self-referential accuracy number
    you can actually report on real data where no ground truth exists.
    """
    n = len(ptsA)
    if n < 10:
        return float('nan')
    rng = np.random.default_rng(seed)
    sel = rng.choice(n, min(n, max_pts), replace=False)
    errs = []
    for i in sel:
        m = np.ones(n, bool)
        m[i] = False
        H = fit_fn(ptsA[m], ptsB[m])
        if H is None:
            continue
        e = apply_H(H, ptsA[i:i + 1]) - ptsB[i:i + 1]
        errs.append(float(np.hypot(e[0, 0], e[0, 1])))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float('nan')
