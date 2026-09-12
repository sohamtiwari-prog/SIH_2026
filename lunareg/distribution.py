"""
distribution.py — making tie points spatially uniform, and proving that they are.

The problem statement asks for match points "maintaining uniform distribution
across the images". This is not cosmetic. Tie points cluster on high-texture
terrain (fresh crater fields, rugged rims) and vanish on smooth mare. A model
fitted to a clustered point set is well-constrained where the points are and
extrapolates badly everywhere else, so the reported RMSE — computed on those
same clustered points — looks excellent while the actual registration drifts
across the empty regions.

So we do two things:
  * ENFORCE spread: block-adaptive selection + adaptive non-maximal suppression.
  * MEASURE spread: four complementary statistics, because any single one can
    be gamed.

Metrics
-------
Grid coverage       fraction of KxK cells holding at least one point.
                    Directly answers "are there gaps?".
Normalised entropy  H(cell counts) / log(K^2). Penalises a few over-full cells
                    even when coverage is complete.
Clark-Evans R       mean-NN-distance / expected-NN-distance for a Poisson
                    process of the same intensity. R<1 clustered, R=1 random,
                    R>1 dispersed. Edge-corrected (Donnelly).
Ripley K deviation  departure from complete spatial randomness across radii;
                    catches multi-scale clustering the others miss.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# Enforcement
# --------------------------------------------------------------------------

def block_select(pts: np.ndarray, scores: np.ndarray, shape, grid: int = 8,
                 per_cell: int = 4) -> np.ndarray:
    """
    Keep the top `per_cell` points by score inside each of grid x grid blocks.

    Cheap, deterministic, and it guarantees a hard ceiling on how much any one
    textured region can dominate the solution.
    """
    h, w = shape
    if len(pts) == 0:
        return np.zeros(0, int)
    cx = np.clip((pts[:, 0] / w * grid).astype(int), 0, grid - 1)
    cy = np.clip((pts[:, 1] / h * grid).astype(int), 0, grid - 1)
    cell = cy * grid + cx
    keep = []
    for c in np.unique(cell):
        idx = np.nonzero(cell == c)[0]
        idx = idx[np.argsort(-scores[idx])][:per_cell]
        keep.append(idx)
    return np.concatenate(keep) if keep else np.zeros(0, int)


def anms(pts: np.ndarray, scores: np.ndarray, n_keep: int,
         robust: float = 1.11) -> np.ndarray:
    """
    Adaptive non-maximal suppression (Brown, Szeliski & Winder 2005).

    Each point gets a suppression radius = distance to the nearest point that is
    meaningfully stronger. Keeping the largest radii yields points that are both
    strong AND spread out, without imposing an arbitrary grid.
    """
    n = len(pts)
    if n == 0:
        return np.zeros(0, int)
    if n <= n_keep:
        return np.arange(n)

    order = np.argsort(-scores)
    p = pts[order]
    s = scores[order]
    radii = np.full(n, np.inf)
    tree = cKDTree(p)

    for i in range(1, n):
        # candidates that are stronger than p[i] by the robustness factor
        stronger = np.nonzero(s[:i] > s[i] * robust)[0]
        if len(stronger) == 0:
            continue
        d = np.linalg.norm(p[stronger] - p[i], axis=1)
        radii[i] = d.min()
    del tree

    sel = np.argsort(-radii)[:n_keep]
    return order[sel]


def enforce_uniform(pts_src, pts_ref, scores, shape, target: int = 400,
                    grid: int = 8) -> np.ndarray:
    """Two-stage: block cap, then ANMS down to the target count."""
    if len(pts_src) == 0:
        return np.zeros(0, int)
    per_cell = max(2, int(np.ceil(target / (grid * grid) * 1.8)))
    k1 = block_select(pts_ref, scores, shape, grid=grid, per_cell=per_cell)
    if len(k1) <= target:
        return k1
    k2 = anms(pts_ref[k1], scores[k1], target)
    return k1[k2]


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def grid_coverage(pts: np.ndarray, shape, grid: int = 8) -> float:
    if len(pts) == 0:
        return 0.0
    h, w = shape
    cx = np.clip((pts[:, 0] / w * grid).astype(int), 0, grid - 1)
    cy = np.clip((pts[:, 1] / h * grid).astype(int), 0, grid - 1)
    return float(len(np.unique(cy * grid + cx)) / (grid * grid))


def cell_entropy(pts: np.ndarray, shape, grid: int = 8) -> float:
    if len(pts) == 0:
        return 0.0
    h, w = shape
    cx = np.clip((pts[:, 0] / w * grid).astype(int), 0, grid - 1)
    cy = np.clip((pts[:, 1] / h * grid).astype(int), 0, grid - 1)
    counts = np.bincount(cy * grid + cx, minlength=grid * grid).astype(float)
    p = counts / counts.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(grid * grid))


def clark_evans(pts: np.ndarray, shape) -> float:
    """
    Clark-Evans nearest-neighbour index with Donnelly edge correction.

    R = observed mean NN distance / expected under CSR.
    """
    n = len(pts)
    if n < 4:
        return float('nan')
    h, w = shape
    A = float(h * w)
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    r_obs = float(d[:, 1].mean())
    perim = 2.0 * (h + w)
    r_exp = 0.5 * np.sqrt(A / n) + 0.0514 * perim / n + 0.041 * perim / (n ** 1.5)
    return r_obs / max(r_exp, 1e-9)


def ripley_deviation(pts: np.ndarray, shape, n_r: int = 12) -> float:
    """
    Mean absolute deviation of the L-function from CSR, normalised by scene size.

    L(r) = sqrt(K(r)/pi). Under complete spatial randomness L(r) = r, so
    mean|L(r) - r| / max_r is 0 for ideal uniformity and grows with clustering.
    """
    n = len(pts)
    if n < 6:
        return float('nan')
    h, w = shape
    A = float(h * w)
    tree = cKDTree(pts)
    rmax = 0.25 * min(h, w)
    rs = np.linspace(rmax / n_r, rmax, n_r)
    lam = n / A
    dev = []
    for r in rs:
        cnt = tree.query_ball_point(pts, r, return_length=True) - 1
        K = cnt.mean() / lam
        L = np.sqrt(max(K, 0) / np.pi)
        dev.append(abs(L - r))
    return float(np.mean(dev) / rmax)


def uniformity_report(pts: np.ndarray, shape, grid: int = 8) -> dict:
    cov = grid_coverage(pts, shape, grid)
    ent = cell_entropy(pts, shape, grid)
    ce = clark_evans(pts, shape)
    rip = ripley_deviation(pts, shape)

    # composite in [0,1]. Clark-Evans is folded so that both clustering (R<1)
    # and unnatural lattice regularity (R>>1) are penalised relative to R~1.3,
    # which is what good ANMS output looks like.
    ce_score = float(np.exp(-((np.nan_to_num(ce, nan=0.0) - 1.30) ** 2) / (2 * 0.45 ** 2)))
    rip_score = float(np.exp(-np.nan_to_num(rip, nan=1.0) / 0.12))
    composite = 0.35 * cov + 0.30 * ent + 0.20 * ce_score + 0.15 * rip_score
    return dict(coverage=cov, entropy=ent, clark_evans=ce,
                ripley_dev=rip, uniformity=float(composite), n=int(len(pts)))
