#!/usr/bin/env python3
"""
lunareg_all_in_one.py — single-file build of the lunareg lunar-image
registration pipeline (SIH 2026, ISRO PS 26166).

This file is a mechanical concatenation of every module in the `lunareg/`
package (see that directory for the same code split up, with per-module
docstrings explaining the reasoning behind each stage) into one script that
runs with nothing but `pip install -r requirements.txt` and no package
install step. It exists for quick demos / environments where installing a
local package is inconvenient; `lunareg/` + `pip install -e .` remains the
source of truth for development. Regenerate it with
`python scripts/build_single_file.py` after changing anything under lunareg/.

Usage
-----
    python lunareg_all_in_one.py register --src A.xml --ref B.tif --method hybrid
    python lunareg_all_in_one.py experiments --out-dir outputs
    python lunareg_all_in_one.py analyse     --out-dir outputs
    python lunareg_all_in_one.py dashboard   --out-dir outputs

Run `python lunareg_all_in_one.py -h` for the full option list.
"""

from __future__ import annotations


# ==========================================================================
# photometric.py
# ==========================================================================

"""
photometric.py — turning illumination-dependent images into illumination-invariant ones.

The core problem
----------------
On an airless body the ONLY thing that changes between two views of the same
terrain is the solar geometry. A crater lit from the east has a bright west
wall and a shadowed east wall; lit from the west, the pattern inverts. Intensity
gradients reverse by 180 degrees.

SIFT descriptors are gradient-orientation histograms, so a 180-degree gradient
flip moves every bin. This is precisely why SIFT collapses on lunar pairs with
large solar azimuth differences — it is not a tuning problem, it is structural.

What survives illumination change
---------------------------------
Fourier PHASE, not amplitude. At a step or ridge, the local frequency
components all arrive in phase. That congruency of phase is a property of the
feature's geometry, and it is invariant to contrast, brightness and — critically
— to the sign of the gradient. Phase congruency (Kovesi 1999) measures it.

We build:
  * PC maximum moment  -> edge strength (crater rims, ridges, scarps)
  * PC minimum moment  -> corner strength (rim junctions, boulder fields)
  * MIM                -> Maximum Index Map, the orientation channel with the
                          largest log-Gabor response. This is the descriptor
                          substrate for RIFT (Li et al. 2020).
"""


import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# Log-Gabor filter bank
# --------------------------------------------------------------------------

def _lowpass_butterworth(shape, cutoff: float, n: int):
    rows, cols = shape
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    x = (x - cols // 2) / cols
    y = (y - rows // 2) / rows
    radius = np.sqrt(x ** 2 + y ** 2)
    return np.fft.ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * n)))


def _grids(shape):
    rows, cols = shape
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    x = (x - cols // 2) / cols
    y = (y - rows // 2) / rows
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    radius[0, 0] = 1.0
    return radius, theta


def phase_congruency(img: np.ndarray, nscale: int = 4, norient: int = 6,
                     min_wavelength: float = 3.0, mult: float = 2.1,
                     sigma_onf: float = 0.55, k: float = 2.0,
                     cutoff: float = 0.5, g: float = 10.0,
                     noise_method: float = -1):
    """
    Kovesi phase congruency.

    Returns dict with:
      M     : maximum moment of the PC covariance  (edge strength, [0,1])
      m     : minimum moment                        (corner strength, [0,1])
      ori   : dominant orientation (radians)
      MIM   : maximum index map, int in [0, norient)
      A     : per-orientation summed amplitude, shape (norient, H, W)
    """
    img = np.asarray(img, dtype=np.float64)
    rows, cols = img.shape
    IMG = np.fft.fft2(img)

    radius, theta = _grids((rows, cols))
    sintheta, costheta = np.sin(theta), np.cos(theta)
    lp = _lowpass_butterworth((rows, cols), 0.45, 15)

    # radial log-Gabor components
    log_gabor = []
    for s in range(nscale):
        wavelength = min_wavelength * mult ** s
        fo = 1.0 / wavelength
        lg = np.exp(-(np.log(radius / fo) ** 2) / (2 * np.log(sigma_onf) ** 2))
        lg = lg * lp
        lg[0, 0] = 0.0
        log_gabor.append(lg)

    theta_sigma = np.pi / norient / 1.2

    cov_x2 = np.zeros((rows, cols))
    cov_y2 = np.zeros((rows, cols))
    cov_xy = np.zeros((rows, cols))
    A_all = np.zeros((norient, rows, cols))
    PC_all = np.zeros((norient, rows, cols))
    eps = 1e-8

    for o in range(norient):
        angl = o * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread = np.exp(-(dtheta ** 2) / (2 * theta_sigma ** 2))

        sumE = np.zeros((rows, cols))
        sumO = np.zeros((rows, cols))
        sumAn = np.zeros((rows, cols))
        maxAn = None
        tau = None

        for s in range(nscale):
            filt = log_gabor[s] * spread
            resp = np.fft.ifft2(IMG * filt)
            E, O = resp.real, resp.imag
            An = np.abs(resp)
            sumE += E
            sumO += O
            sumAn += An
            if s == 0:
                # Rayleigh noise estimate from the finest scale
                tau = np.median(An) / np.sqrt(np.log(4.0))
                maxAn = An
            else:
                maxAn = np.maximum(maxAn, An)

        XEnergy = np.sqrt(sumE ** 2 + sumO ** 2) + eps
        MeanE, MeanO = sumE / XEnergy, sumO / XEnergy

        # recompute the phase-deviation weighted energy
        Energy = np.zeros((rows, cols))
        for s in range(nscale):
            filt = log_gabor[s] * spread
            resp = np.fft.ifft2(IMG * filt)
            E, O = resp.real, resp.imag
            Energy += E * MeanE + O * MeanO - np.abs(E * MeanO - O * MeanE)

        if noise_method < 0:
            totalTau = tau * (1 - (1 / mult) ** nscale) / (1 - (1 / mult))
            EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)
            EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2)
            T = EstNoiseEnergyMean + k * EstNoiseEnergySigma
        else:
            T = noise_method
        Energy = np.maximum(Energy - T, 0.0)

        # frequency-spread weighting: reject responses from a single scale
        width = (sumAn / (maxAn + eps) - 1.0) / (nscale - 1)
        weight = 1.0 / (1.0 + np.exp(g * (cutoff - width)))

        PC = weight * Energy / (sumAn + eps)
        PC_all[o] = PC
        A_all[o] = sumAn

        cov_x2 += (PC * np.cos(angl)) ** 2
        cov_y2 += (PC * np.sin(angl)) ** 2
        cov_xy += (PC * np.cos(angl)) * (PC * np.sin(angl))

    cov_x2 /= (norient / 2.0)
    cov_y2 /= (norient / 2.0)
    cov_xy *= 2.0 / (norient / 2.0)

    denom = np.sqrt(cov_xy ** 2 + (cov_x2 - cov_y2) ** 2) + eps
    M = (cov_y2 + cov_x2 + denom) / 2.0
    m = (cov_y2 + cov_x2 - denom) / 2.0
    ori = np.arctan2(cov_y2 - m, cov_xy) / 2.0

    MIM = np.argmax(A_all, axis=0).astype(np.int32)

    return dict(M=_norm01(M), m=_norm01(m), ori=ori, MIM=MIM, A=A_all,
                norient=norient)


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)


# --------------------------------------------------------------------------
# Classical photometric normalisation (uses metadata when available)
# --------------------------------------------------------------------------

def lommel_seeliger_correct(img: np.ndarray, inc_deg: float, emi_deg: float,
                            L: float = 0.85) -> np.ndarray:
    """
    Divide out the first-order photometric function using known illumination
    angles from the PDS label. Removes the large-scale brightness ramp across a
    scene; it cannot remove cast shadows, which is why phase congruency is still
    needed downstream.
    """
    i, e = np.deg2rad(inc_deg), np.deg2rad(emi_deg)
    mu0, mu = max(np.cos(i), 1e-3), max(np.cos(e), 1e-3)
    f = mu0 * (2 * L * mu0 / (mu0 + mu) + (1 - L))
    return img / max(f, 1e-6)


def clahe_normalise(img: np.ndarray, clip: float = 2.5, tiles: int = 8) -> np.ndarray:
    """Local contrast equalisation — cheap partial illumination invariance."""
    import cv2
    u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return c.apply(u8).astype(np.float32) / 255.0


def shadow_fraction(img: np.ndarray, thresh_pct: float = 8.0) -> float:
    """Fraction of pixels in deep shadow — a usable-signal diagnostic."""
    t = np.percentile(img, thresh_pct)
    return float((img <= max(t, 1e-6)).mean())


def gradient_flip_index(a: np.ndarray, b: np.ndarray) -> float:
    """
    Diagnostic: correlation of gradient orientation fields between two images.

    Near +1  -> same illumination regime, SIFT will work.
    Near -1  -> gradients inverted by shadow flip, SIFT will fail.
    This quantifies WHY a pair is hard, before you try to register it.
    """
    ax, ay = np.gradient(ndimage.gaussian_filter(a, 2.0))
    bx, by = np.gradient(ndimage.gaussian_filter(b, 2.0))
    na = np.hypot(ax, ay) + 1e-9
    nb = np.hypot(bx, by) + 1e-9
    w = (na * nb)
    cos = (ax * bx + ay * by) / w
    return float(np.average(cos, weights=w))

# ==========================================================================
# metrics.py
# ==========================================================================

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

# ==========================================================================
# distribution.py
# ==========================================================================

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

# ==========================================================================
# matching.py
# ==========================================================================

"""
matching.py — descriptor matching, robust geometry, sub-pixel refinement.

Pipeline order matters:
  1. mutual nearest neighbour + Lowe ratio  -> candidate correspondences
  2. MAGSAC++ on a progressive model ladder -> reject outliers
  3. per-point sub-pixel refinement          -> push residuals below 1 px

Step 3 is where the "sub-pixel accuracy" requirement is actually met. Descriptor
matching localises to roughly +/-1 px at best because keypoints sit on a discrete
grid. Upsampled-DFT phase correlation on a small window around each surviving
match recovers the fractional part, and it does so using the normalised
cross-power spectrum, which discards amplitude and is therefore itself
illumination-robust.
"""


import cv2
import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# Descriptor matching
# --------------------------------------------------------------------------

def match_descriptors(desA: np.ndarray, desB: np.ndarray, binary: bool = False,
                      ratio: float = 0.85, mutual: bool = True) -> np.ndarray:
    """Return index pairs [M,2] into (A, B)."""
    if desA is None or desB is None or len(desA) < 2 or len(desB) < 2:
        return np.zeros((0, 2), int)

    norm = cv2.NORM_HAMMING if binary else cv2.NORM_L2
    bf = cv2.BFMatcher(norm)

    knn = bf.knnMatch(desA, desB, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))
    if not good:
        return np.zeros((0, 2), int)
    good = np.array(good, int)

    if mutual:
        knn_r = bf.knnMatch(desB, desA, k=1)
        back = {p[0].queryIdx: p[0].trainIdx for p in knn_r if len(p)}
        keep = [i for i, (a, b) in enumerate(good) if back.get(b, -1) == a]
        good = good[keep] if keep else np.zeros((0, 2), int)
    return good


# --------------------------------------------------------------------------
# Robust geometric estimation
# --------------------------------------------------------------------------

MODEL_LADDER = ('similarity', 'affine', 'homography')


def estimate_model(ptsA: np.ndarray, ptsB: np.ndarray, model: str = 'homography',
                   thresh: float = 3.0, conf: float = 0.9995, iters: int = 20000):
    """
    Robust fit of A -> B. Returns (H 3x3, inlier_mask bool[N]) or (None, None).

    MAGSAC++ is used where OpenCV supports it: it marginalises over the inlier
    threshold instead of committing to one, which matters here because the
    correct threshold varies with the scale ratio of the pair.
    """
    n = len(ptsA)
    need = {'similarity': 3, 'affine': 3, 'homography': 4}[model]
    if n < need:
        return None, None
    A = np.ascontiguousarray(ptsA, np.float32).reshape(-1, 1, 2)
    B = np.ascontiguousarray(ptsB, np.float32).reshape(-1, 1, 2)

    if model == 'homography':
        H, mask = cv2.findHomography(A, B, cv2.USAC_MAGSAC, thresh,
                                     maxIters=iters, confidence=conf)
    elif model == 'affine':
        M, mask = cv2.estimateAffine2D(A, B, method=cv2.USAC_MAGSAC,
                                       ransacReprojThreshold=thresh,
                                       maxIters=iters, confidence=conf)
        H = np.vstack([M, [0, 0, 1.0]]) if M is not None else None
    else:
        M, mask = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC,
                                              ransacReprojThreshold=thresh,
                                              maxIters=iters, confidence=conf)
        H = np.vstack([M, [0, 0, 1.0]]) if M is not None else None

    if H is None or mask is None:
        return None, None
    return H.astype(np.float64), mask.ravel().astype(bool)


def estimate_progressive(ptsA, ptsB, thresh: float = 3.0, min_inliers: int = 12):
    """
    Climb the model ladder. Start with the most constrained model that a lunar
    pair usually satisfies (similarity: scale + rotation + shift), then relax to
    affine, then homography — accepting the richer model only if it keeps enough
    inliers. This prevents a homography from folding the image to fit noise when
    the true relation is a simple scale change, which is the standard failure
    mode on low-texture mare.
    """
    best = (None, None, None)
    for model in MODEL_LADDER:
        H, mask = estimate_model(ptsA, ptsB, model, thresh)
        if H is None or mask is None:
            continue
        if mask.sum() < min_inliers:
            continue
        if best[0] is None or mask.sum() >= best[1].sum() * 0.92:
            best = (H, mask, model)
    return best


# --------------------------------------------------------------------------
# Sub-pixel refinement
# --------------------------------------------------------------------------

def _upsampled_dft_shift(a: np.ndarray, b: np.ndarray, upsample: int = 50,
                         max_shift: float = 3.0):
    """
    Sub-pixel shift of `b` relative to `a`: returns (dr, dc, psr) such that
    shifting `b` by (dr, dc) aligns it with `a`.

    Uses the normalised cross-power spectrum, so amplitude is divided out and
    only phase survives — the same invariance argument as phase congruency,
    applied to alignment rather than detection.

    The upsampled refinement is delegated to skimage's Guizar-Sicairos
    implementation. A hand-rolled version of the matrix-multiply DFT here was
    biased by ~0.75 px even at zero shift, which is fatal when the deliverable
    is sub-pixel accuracy — the bias is invisible unless you unit-test the
    estimator against known synthetic shifts, so it is worth doing that.
    """
    from skimage.registration import phase_cross_correlation

    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return None
    if min(a.shape) < 6:
        return None
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return None
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    aw, bw = a * win, b * win

    # peak-to-sidelobe ratio from the coarse correlation surface
    FA, FB = np.fft.fft2(aw), np.fft.fft2(bw)
    R = FA * np.conj(FB)
    R /= np.abs(R) + 1e-12
    mag = np.abs(np.fft.ifft2(R))
    peak = np.unravel_index(np.argmax(mag), mag.shape)
    mask = np.ones_like(mag, bool)
    rr = 3
    mask[max(0, peak[0] - rr):peak[0] + rr + 1,
         max(0, peak[1] - rr):peak[1] + rr + 1] = False
    side = mag[mask]
    psr = float((mag[peak] - side.mean()) / (side.std() + 1e-12))

    shift, _, _ = phase_cross_correlation(aw, bw, upsample_factor=upsample,
                                          normalization='phase')
    dr, dc = float(shift[0]), float(shift[1])
    if not (np.isfinite(dr) and np.isfinite(dc)):
        return None
    if abs(dr) > max_shift or abs(dc) > max_shift:
        return None
    return dr, dc, psr


def refine_subpixel(imgA: np.ndarray, imgB: np.ndarray, ptsA: np.ndarray,
                    ptsB: np.ndarray, H: np.ndarray, win: int = 32,
                    upsample: int = 50, max_shift: float = 2.5,
                    use_pc: bool = True):
    """
    For each correspondence, cut a window from A (warped into B's frame) and the
    matching window from B, and estimate the residual fractional shift.

    Returns (refined_ptsB, delta [N,2], quality [N], ok mask).
    """

    hB, wB = imgB.shape
    Awarp = cv2.warpPerspective(imgA, H, (wB, hB), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REFLECT)
    if use_pc:
        # phase congruency of both, so the correlation is illumination-invariant
        Awarp = phase_congruency(Awarp, nscale=3, norient=4)['M']
        B = phase_congruency(imgB, nscale=3, norient=4)['M']
    else:
        B = imgB

    half = win // 2
    out = ptsB.copy().astype(np.float64)
    delta = np.zeros((len(ptsB), 2))
    qual = np.zeros(len(ptsB))
    ok = np.zeros(len(ptsB), bool)

    # project A points into B's frame to know where to cut
    P = np.hstack([ptsA, np.ones((len(ptsA), 1))]).T
    Q = H @ P
    Q = (Q[:2] / Q[2]).T

    for i, (qx, qy) in enumerate(Q):
        x, y = int(round(qx)), int(round(qy))
        if x - half < 0 or y - half < 0 or x + half >= wB or y + half >= hB:
            continue
        pa = Awarp[y - half:y + half, x - half:x + half]
        pb = B[y - half:y + half, x - half:x + half]
        res = _upsampled_dft_shift(pb, pa, upsample=upsample, max_shift=max_shift)
        if res is None:
            continue
        dr, dc, pv = res
        if not np.isfinite(pv) or abs(dr) > max_shift or abs(dc) > max_shift:
            continue
        # pa (warped source) shifted by (dr, dc) matches pb (reference), so the
        # reference location truly corresponding to this source point is offset
        # by +(dc, dr) from the current estimate.
        out[i] = (ptsB[i][0] + dc, ptsB[i][1] + dr)
        delta[i] = (dc, dr)
        qual[i] = pv
        ok[i] = True
    return out, delta, qual, ok

# ==========================================================================
# features.py
# ==========================================================================

"""
features.py — detectors and descriptors.

Method inventory (all exposed through `detect_and_describe`):

  sift        Baseline. Fails hard under solar azimuth flip, by construction.
  pcsift      SIFT run on the phase-congruency map. One-line change, large gain.
  rift        Phase-congruency detector + Maximum-Index-Map descriptor.
              Illumination-invariant by design.
  akaze       Nonlinear scale space; a stronger classical baseline.
  orb         Fast binary baseline; included to show the speed/accuracy floor.

Rotation invariance for RIFT is handled two ways at once:
  1. the sampling grid is rotated to a dominant orientation estimated from the
     phase-congruency orientation field, and
  2. the MIM index bins are cyclically shifted by the same amount, because
     rotating the image permutes which log-Gabor orientation channel wins.
Doing only (2) — as the original RIFT does — breaks past ~30 degrees.
"""


import cv2
import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def pc_keypoints(pc: dict, n_max: int = 3000, min_dist: int = 4,
                 corner_weight: float = 0.65) -> np.ndarray:
    """
    Keypoints from phase congruency moments.

    score = w * minimum-moment (corner-like) + (1-w) * maximum-moment (edge-like)

    Crater rims are edges; rim intersections, central peaks and boulders are
    corners. Weighting toward corners gives better-localised points, but pure
    corner response is sparse on smooth mare, so we blend.
    """
    score = corner_weight * pc['m'] + (1.0 - corner_weight) * pc['M']
    score = ndimage.gaussian_filter(score, 1.0)

    mx = ndimage.maximum_filter(score, size=2 * min_dist + 1)
    peaks = (score == mx) & (score > np.percentile(score, 80))
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return np.zeros((0, 3), np.float32)
    vals = score[ys, xs]
    order = np.argsort(-vals)[:n_max]
    return np.stack([xs[order], ys[order], vals[order]], 1).astype(np.float32)


def _dominant_orientation(pc: dict, x: float, y: float, radius: int = 12) -> float:
    """Amplitude-weighted circular mean of the PC orientation field."""
    h, w = pc['ori'].shape
    x0, x1 = int(max(0, x - radius)), int(min(w, x + radius + 1))
    y0, y1 = int(max(0, y - radius)), int(min(h, y + radius + 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    o = pc['ori'][y0:y1, x0:x1]
    m = pc['M'][y0:y1, x0:x1]
    # orientation is pi-periodic -> double the angle before averaging
    s = np.sum(m * np.sin(2 * o))
    c = np.sum(m * np.cos(2 * o))
    return float(0.5 * np.arctan2(s, c))


# --------------------------------------------------------------------------
# RIFT-style descriptor on the Maximum Index Map
# --------------------------------------------------------------------------

def sun_bin_shift(d_azimuth_deg: float, norient: int = 6) -> int:
    """
    Predict the MIM orientation-bin shift induced by a solar azimuth change.

    Shadow edges run perpendicular to the illumination direction, so the winning
    log-Gabor orientation channel rotates WITH the sun. Because orientation is
    pi-periodic, a 180 degree azimuth change maps back to zero shift — which is
    why registration difficulty peaks near 90 degrees rather than at 180, and
    why a naive "larger azimuth difference is harder" assumption is wrong.

        shift = round( (d_az mod 180) / 180 * norient )  mod norient
    """
    d = abs(d_azimuth_deg) % 180.0
    return int(round(d / 180.0 * norient)) % norient


def mim_descriptor_bank(pc: dict, kps: np.ndarray, shifts, patch: int = 72,
                        grid: int = 6, rotate: bool = True):
    """
    Build descriptors for several cyclic MIM shifts while sampling the patch
    lattice only once. Sampling dominates the cost, so N shifts cost far less
    than N independent descriptor passes.

    Returns (list_of_descriptor_arrays, kept_kps).
    """
    MIM = pc['MIM']
    norient = pc['norient']
    h, w = MIM.shape
    half = patch // 2

    lin = (np.arange(patch) - half + 0.5)
    gx, gy = np.meshgrid(lin, lin)
    cell = patch / grid
    cell_ix = np.clip(((gx + half) // cell).astype(int), 0, grid - 1)
    cell_iy = np.clip(((gy + half) // cell).astype(int), 0, grid - 1)
    cell_id = (cell_iy * grid + cell_ix).ravel()

    sampled, keep = [], []
    for kp in kps:
        x, y = float(kp[0]), float(kp[1])
        if x < half * 0.6 or y < half * 0.6 or x > w - half * 0.6 or y > h - half * 0.6:
            continue
        th = _dominant_orientation(pc, x, y) if rotate else 0.0
        c, s = np.cos(th), np.sin(th)
        sx = x + c * gx - s * gy
        sy = y + s * gx + c * gy
        vals = ndimage.map_coordinates(MIM, [sy, sx], order=0, mode='nearest').ravel()
        if rotate:
            vals = (vals - int(np.round(th / np.pi * norient))) % norient
        sampled.append(vals)
        keep.append(kp)

    D = grid * grid * norient
    if not sampled:
        return [np.zeros((0, D), np.float32) for _ in shifts], np.zeros((0, 3), np.float32)

    S = np.stack(sampled)
    banks = []
    for sh in shifts:
        v = (S - sh) % norient
        idx = cell_id[None, :] * norient + v
        hist = np.stack([np.bincount(row, minlength=D) for row in idx]).astype(np.float32)
        n = np.linalg.norm(hist, axis=1, keepdims=True)
        hist = hist / np.maximum(n, 1e-6)
        hist = np.clip(hist, 0, 0.2)
        hist = hist / np.maximum(np.linalg.norm(hist, axis=1, keepdims=True), 1e-6)
        banks.append(hist.astype(np.float32))
    return banks, np.stack(keep).astype(np.float32)


def mim_descriptor(pc: dict, kps: np.ndarray, patch: int = 72, grid: int = 6,
                   rotate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Descriptor = per-cell histogram of MIM indices over a grid x grid layout.

    Returns (descriptors [N, grid*grid*norient] float32 L2-normalised, kept_kps).
    """
    MIM = pc['MIM']
    norient = pc['norient']
    h, w = MIM.shape
    half = patch // 2

    # sampling lattice in patch-local coordinates
    lin = (np.arange(patch) - half + 0.5)
    gx, gy = np.meshgrid(lin, lin)
    cell = patch / grid
    cell_ix = np.clip(((gx + half) // cell).astype(int), 0, grid - 1)
    cell_iy = np.clip(((gy + half) // cell).astype(int), 0, grid - 1)
    cell_id = cell_iy * grid + cell_ix

    descs, keep = [], []
    for kp in kps:
        x, y = float(kp[0]), float(kp[1])
        if x < half * 0.6 or y < half * 0.6 or x > w - half * 0.6 or y > h - half * 0.6:
            continue
        th = _dominant_orientation(pc, x, y) if rotate else 0.0
        c, s = np.cos(th), np.sin(th)
        sx = x + c * gx - s * gy
        sy = y + s * gx + c * gy

        vals = ndimage.map_coordinates(MIM, [sy, sx], order=0, mode='nearest')
        if rotate:
            # rotating the patch permutes the winning orientation channel
            shift = int(np.round(th / np.pi * norient)) % norient
            vals = (vals - shift) % norient

        idx = cell_id.ravel() * norient + vals.ravel()
        hist = np.bincount(idx, minlength=grid * grid * norient).astype(np.float32)

        # SIFT-style two-stage normalisation: L2, clip, L2
        n = np.linalg.norm(hist)
        if n < 1e-6:
            continue
        hist /= n
        hist = np.clip(hist, 0, 0.2)
        hist /= max(np.linalg.norm(hist), 1e-6)
        descs.append(hist)
        keep.append(kp)

    if not descs:
        return np.zeros((0, grid * grid * norient), np.float32), np.zeros((0, 3), np.float32)
    return np.stack(descs).astype(np.float32), np.stack(keep).astype(np.float32)


# --------------------------------------------------------------------------
# Unified entry point
# --------------------------------------------------------------------------

def _to_u8(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [0.5, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)


def detect_and_describe(img: np.ndarray, method: str = 'rift',
                        n_max: int = 3000, pc_cache: dict | None = None):
    """
    Returns (keypoints [N,2] float32 xy, descriptors [N,D], is_binary, pc_dict).
    """
    method = method.lower()

    if method in ('sift', 'orb', 'akaze'):
        u8 = _to_u8(img)
        if method == 'sift':
            det = cv2.SIFT_create(nfeatures=n_max)
        elif method == 'orb':
            det = cv2.ORB_create(nfeatures=n_max, fastThreshold=7)
        else:
            det = cv2.AKAZE_create()
        kp, des = det.detectAndCompute(u8, None)
        if des is None or len(kp) == 0:
            return np.zeros((0, 2), np.float32), None, method == 'orb', None
        pts = np.array([k.pt for k in kp], np.float32)
        return pts, des, method in ('orb', 'akaze'), None

    pc = pc_cache if pc_cache is not None else phase_congruency(img)

    if method == 'pcsift':
        u8 = _to_u8(pc['M'])
        det = cv2.SIFT_create(nfeatures=n_max)
        kp, des = det.detectAndCompute(u8, None)
        if des is None or len(kp) == 0:
            return np.zeros((0, 2), np.float32), None, False, pc
        pts = np.array([k.pt for k in kp], np.float32)
        return pts, des, False, pc

    if method == 'rift':
        kps = pc_keypoints(pc, n_max=n_max)
        des, kept = mim_descriptor(pc, kps)
        if len(kept) == 0:
            return np.zeros((0, 2), np.float32), None, False, pc
        return kept[:, :2].astype(np.float32), des, False, pc

    raise ValueError(f'unknown method: {method}')

# ==========================================================================
# dense.py
# ==========================================================================

"""
dense.py — area-based tie-point extraction on phase-congruency maps.

Why this module exists
----------------------
The diagnostics in the experiment log show a clean split:

  * AREA similarity of phase congruency stays around +0.55 to +0.67 even at a
    90-180 degree solar azimuth difference.
  * KEYPOINT repeatability over the same range collapses to ~0.43, and MIM
    agreement to ~0.2-0.3.

So the signal is present, but it is not concentrated at repeatable interest
points. Sparse descriptor matching therefore fails on exactly the pairs we most
need to handle, while area-based matching still works.

This is also how operational planetary pipelines are built: a geometric prior
from SPICE/PDS metadata gets you close, then dense area correlation does the
work. Feature matching is used only to bootstrap the prior when metadata is
absent or untrustworthy.

Three extra benefits fall out for free:
  * uniform distribution is guaranteed by construction — the templates ARE a grid;
  * sub-pixel accuracy comes from the same upsampled-DFT estimator;
  * the correlation is on the normalised cross-power spectrum, so it discards
    amplitude and inherits illumination robustness.
"""


import cv2
import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# Coarse alignment: Fourier-Mellin on phase congruency
# --------------------------------------------------------------------------

def _logpolar_spectrum(img: np.ndarray, n_theta: int = 180, n_rho: int = 128):
    """High-passed log-polar magnitude spectrum — translation invariant."""
    h, w = img.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fftshift(np.abs(np.fft.fft2(img * win)))
    F = np.log1p(F)
    cy, cx = h / 2.0, w / 2.0
    rmax = min(cy, cx)
    rmin = 3.0
    rho = np.exp(np.linspace(np.log(rmin), np.log(rmax), n_rho))
    th = np.linspace(0, np.pi, n_theta, endpoint=False)
    R, T = np.meshgrid(rho, th, indexing='ij')
    ys = cy + R * np.sin(T)
    xs = cx + R * np.cos(T)
    return ndimage.map_coordinates(F, [ys, xs], order=1, mode='nearest'), rho


def estimate_rotation_scale(A: np.ndarray, B: np.ndarray, use_pc: bool = True):
    """
    Estimate (rotation_deg, scale) mapping A onto B, translation-free.

    Runs on phase congruency so that the spectra compare structure rather than
    illumination. Rotation is recovered modulo 180 degrees, which is resolved
    downstream by testing both candidates.
    """
    if use_pc:
        A = phase_congruency(A, nscale=3, norient=6)['M']
        B = phase_congruency(B, nscale=3, norient=6)['M']
    n = min(A.shape + B.shape)
    if n < 16:
        return 0.0, 1.0, 0.0
    n = int(2 ** np.floor(np.log2(n)))
    A = cv2.resize(A, (n, n)); B = cv2.resize(B, (n, n))

    LA, rho = _logpolar_spectrum(A)
    LB, _ = _logpolar_spectrum(B)
    res = _upsampled_dft_shift(LB, LA, upsample=20, max_shift=max(LA.shape))
    if res is None:
        return 0.0, 1.0, 0.0
    d_rho, d_th, q = res
    n_theta = LA.shape[1]
    rot = -d_th / n_theta * 180.0
    log_step = (np.log(rho[-1]) - np.log(rho[0])) / (len(rho) - 1)
    scale = float(np.exp(-d_rho * log_step))
    if not np.isfinite(scale) or not (0.5 < scale < 2.0):
        scale = 1.0
    return float(rot), scale, float(q)


def coarse_translation(A: np.ndarray, B: np.ndarray, use_pc: bool = True):
    """Global sub-pixel translation of A relative to B, on PC maps."""
    if use_pc:
        A = phase_congruency(A, nscale=3, norient=4)['M']
        B = phase_congruency(B, nscale=3, norient=4)['M']
    h = min(A.shape[0], B.shape[0]); w = min(A.shape[1], B.shape[1])
    a, b = A[:h, :w], B[:h, :w]
    res = _upsampled_dft_shift(b, a, upsample=20, max_shift=max(h, w))
    if res is None:
        return 0.0, 0.0, 0.0
    dr, dc, q = res
    # _upsampled_dft_shift(b, a) returns the shift that must be APPLIED TO `a`
    # to align it with `b`. Here a=A, b=B, so translating A by (dc, dr) in (x, y)
    # brings it onto B.
    return float(dc), float(dr), float(q)


def coarse_align(src: np.ndarray, ref: np.ndarray, try_180: bool = True):
    """
    Bootstrap homography src -> ref using Fourier-Mellin + phase correlation.

    Used when no reliable metadata prior exists. Returns a 3x3 similarity.
    """
    best = (np.eye(3), -np.inf)
    rot, scale, _ = estimate_rotation_scale(src, ref)
    cands = [rot, rot + 180.0] if try_180 else [rot]
    for rr in cands:
        M = cv2.getRotationMatrix2D((src.shape[1] / 2, src.shape[0] / 2), rr, scale)
        H0 = np.vstack([M, [0, 0, 1.0]])
        warped = cv2.warpPerspective(src, H0, (ref.shape[1], ref.shape[0]),
                                     flags=cv2.INTER_CUBIC)
        dx, dy, q = coarse_translation(warped, ref)
        H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]]) @ H0
        if q > best[1]:
            best = (H, q)
    return best[0]


# --------------------------------------------------------------------------
# Dense grid matching
# --------------------------------------------------------------------------

def grid_tiepoints(src: np.ndarray, ref: np.ndarray, H0: np.ndarray,
                   grid: int = 14, win: int = 48, search: int = 20,
                   upsample: int = 50, min_peak: float = 6.0,
                   min_texture: float = 0.012, use_pc: bool = True):
    """
    Extract tie points on a regular grid in the REFERENCE frame.

    For each grid node we cut a `win`-sized template from the reference and the
    corresponding patch from the prior-warped source, then estimate their
    residual shift to sub-pixel precision. Nodes are rejected when the local
    texture is too weak to localise (flat mare) or the correlation peak is not
    sharp enough.

    Returns (pts_src_original_frame [N,2], pts_ref [N,2], quality [N]).
    """
    hR, wR = ref.shape
    warped = cv2.warpPerspective(src, H0, (wR, hR), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REFLECT)
    valid = cv2.warpPerspective(np.ones_like(src), H0, (wR, hR),
                                flags=cv2.INTER_NEAREST) > 0.5

    if use_pc:
        A = phase_congruency(warped, nscale=4, norient=6)['M']
        B = phase_congruency(ref, nscale=4, norient=6)['M']
    else:
        A, B = warped, ref

    half = win // 2
    pad = half + search
    # The reverse direction of a cycle check can shrink an image below the
    # correlation window, which produced zero-length FFT axes rather than a
    # clean "no tie points" result. Shrink the window to fit, and give up only
    # when even a minimal window will not.
    if min(hR, wR) < 2 * pad + 4:
        win = max(12, int((min(hR, wR) - 8) * 0.5) // 2 * 2)
        search = max(2, min(search, (min(hR, wR) - win - 6) // 2))
        half = win // 2
        pad = half + search
        if min(hR, wR) < 2 * pad + 4 or win < 12:
            return (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
    xs = np.linspace(pad, wR - pad - 1, grid)
    ys = np.linspace(pad, hR - pad - 1, grid)

    pts_ref, deltas, quals = [], [], []
    for y in ys:
        for x in xs:
            xi, yi = int(round(x)), int(round(y))
            sl = (slice(yi - half, yi + half), slice(xi - half, xi + half))
            if not valid[sl].all():
                continue
            pb = B[sl]
            pa = A[sl]
            if pb.std() < min_texture or pa.std() < min_texture:
                continue
            res = _upsampled_dft_shift(pb, pa, upsample=upsample, max_shift=search)
            if res is None:
                continue
            dr, dc, pv = res
            if pv < min_peak or abs(dr) > search or abs(dc) > search:
                continue
            pts_ref.append((x, y))
            deltas.append((dc, dr))
            quals.append(pv)

    if not pts_ref:
        return (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))

    pts_ref = np.array(pts_ref, float)
    deltas = np.array(deltas, float)
    quals = np.array(quals, float)

    # Sign convention, stated explicitly because getting it wrong is silent:
    # delta = (dc, dr) is the shift that must be applied to the warped-source
    # patch to align it with the reference patch. So warped-source content at
    # u corresponds to reference content at u + delta. The warped-source pixel
    # matching reference node p is therefore u = p - delta, NOT p + delta.
    src_in_warped = pts_ref - deltas
    Hi = np.linalg.inv(H0)
    P = np.hstack([src_in_warped, np.ones((len(src_in_warped), 1))]).T
    Q = Hi @ P
    pts_src = (Q[:2] / Q[2]).T
    return pts_src, pts_ref, quals


def dense_register(src: np.ndarray, ref: np.ndarray, H0: np.ndarray,
                   levels=((10, 64, 24), (16, 40, 8), (22, 32, 4)),
                   model: str = 'homography', thresh: float = 2.0):
    """
    Coarse-to-fine dense registration.

    Each level uses a denser grid, a smaller template and a tighter search than
    the last, re-fitting the model in between so the next level starts from a
    better prior. Returns (H, pts_src, pts_ref, quality).
    """
    H = H0.copy()
    ps = pr = np.zeros((0, 2))
    q = np.zeros(0)
    for (g, w, s) in levels:
        ps, pr, q = grid_tiepoints(src, ref, H, grid=g, win=w, search=s)
        if len(ps) < 8:
            break
        Hn, mask, _ = estimate_progressive(ps, pr, thresh)
        if Hn is None:
            Hn, mask = estimate_model(ps, pr, 'affine', thresh)
        if Hn is None:
            break
        H = Hn
        ps, pr, q = ps[mask], pr[mask], q[mask]
    return H, ps, pr, q

# ==========================================================================
# craters.py
# ==========================================================================

"""
craters.py — matching by crater constellations rather than by pixels.

The problem this solves
-----------------------
Every method so far matches APPEARANCE: intensity, phase congruency, gradient
histograms. Appearance is a function of resolution. At a 10:1 GSD ratio the
coarse image simply does not contain the texture the fine image is made of, so
appearance matching has nothing to lock onto — measured: TMC-2 against NAC at
10:1 converges on 0% of runs, IIRS against TC at 8:1 converges but lands 73 px
out.

The insight
-----------
A 500 m crater is 500 m wide in every image ever taken of it. Crater centres are
GEOMETRIC landmarks in metres, not appearance features in pixels. Their relative
arrangement is a rigid property of the surface. So instead of asking "does this
patch look like that patch", ask "does this arrangement of craters match that
arrangement of craters" — exactly how a star tracker identifies its attitude
from the pattern of stars, with no knowledge of what any individual star looks
like.

This is scale-free by construction, illumination-free (a crater rim is circular
whatever the sun does), and it needs no metadata prior at all, which makes it
the natural bootstrap when PDS labels are missing or wrong.

The descriptor
--------------
For each crater i, take its k nearest neighbours and record
    - neighbour distances divided by crater i's own radius   -> scale invariant
    - the turning angles between successive neighbours        -> rotation invariant
Both quantities are dimensionless, so the descriptor of a crater field imaged at
0.5 m/px and at 5 m/px is the same vector.
"""


import cv2
import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def detect_craters(img: np.ndarray, pc: dict | None = None,
                   min_radius_px: int = 5, max_radius_px: int | None = None,
                   max_craters: int = 260, sun_azimuth_deg: float | None = None):
    """
    Detect craters as circles on the phase-congruency map.

    Hough on PC rather than on intensity is deliberate: a crater rim is a closed
    circular ridge in the terrain, and PC responds to that ridge with the same
    sign whichever side the sun is on. Hough on raw intensity finds the
    bright-dark boundary instead, which migrates around the rim as the sun moves.

    When the solar azimuth is known we additionally score each candidate by how
    well its interior brightness splits along the illumination direction — a
    real bowl is bright on the sun-facing wall and dark opposite. That check
    rejects the ring-shaped false positives Hough is prone to.

    Returns array [N, 4] of (x, y, radius_px, score).
    """
    h, w = img.shape
    if max_radius_px is None:
        max_radius_px = int(min(h, w) * 0.16)
    if pc is None:
        pc = phase_congruency(img, nscale=4, norient=6)

    edge = np.clip(pc['M'] * 255, 0, 255).astype(np.uint8)
    edge = cv2.GaussianBlur(edge, (0, 0), 1.1)

    circles = cv2.HoughCircles(
        edge, cv2.HOUGH_GRADIENT, dp=1.0,
        minDist=max(6, min_radius_px * 1.4),
        param1=110, param2=19,
        minRadius=int(min_radius_px), maxRadius=int(max_radius_px))
    if circles is None:
        return np.zeros((0, 4), np.float32)
    circles = circles[0]

    out = []
    for (cx, cy, r) in circles:
        s = _crater_score(img, pc['M'], cx, cy, r, sun_azimuth_deg)
        if s > 0:
            out.append((cx, cy, r, s))
    if not out:
        return np.zeros((0, 4), np.float32)
    out = np.array(out, np.float32)
    return out[np.argsort(-out[:, 3])][:max_craters]


def _crater_score(img, pcM, cx, cy, r, sun_az):
    """Rim strength, plus a bowl-shading check when the sun direction is known."""
    h, w = img.shape
    if cx - r < 1 or cy - r < 1 or cx + r >= w - 1 or cy + r >= h - 1:
        return 0.0
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    rx = np.clip(cx + r * np.cos(th), 0, w - 1)
    ry = np.clip(cy + r * np.sin(th), 0, h - 1)
    rim = pcM[ry.astype(int), rx.astype(int)]
    # a real rim is strong AROUND the whole circle, not on one arc
    rim_score = float(rim.mean() * (1.0 - 0.6 * (rim.std() / (rim.mean() + 1e-6)).clip(0, 1)))

    if sun_az is None:
        return rim_score

    # interior bowl check: sample two half-discs split perpendicular to the sun
    a = np.deg2rad(sun_az)
    ux, uy = np.sin(a), -np.cos(a)          # unit vector toward the sun
    rr = np.linspace(0.15 * r, 0.75 * r, 6)
    pts_l, pts_r = [], []
    for radius in rr:
        for t in th:
            px, py = cx + radius * np.cos(t), cy + radius * np.sin(t)
            side = (px - cx) * ux + (py - cy) * uy
            (pts_l if side > 0 else pts_r).append(
                img[int(np.clip(py, 0, h - 1)), int(np.clip(px, 0, w - 1))])
    if not pts_l or not pts_r:
        return rim_score
    contrast = abs(np.mean(pts_l) - np.mean(pts_r)) / (img.std() + 1e-6)
    return float(rim_score * (0.55 + 0.45 * min(contrast, 1.5) / 1.5))


# --------------------------------------------------------------------------
# Scale- and rotation-invariant constellation descriptor
# --------------------------------------------------------------------------

def constellation_descriptors(craters: np.ndarray, k: int = 6):
    """
    Descriptor per crater from the geometry of its k nearest neighbours.

    Three invariances have to hold simultaneously, and each needs care:

      scale     — distances are divided by the MEDIAN neighbour distance, not by
                  the crater's own fitted radius. Radius estimates from Hough are
                  noisy; the median neighbour distance is a far steadier local
                  length unit.
      rotation  — only angular GAPS between successive neighbours are stored,
                  never absolute bearings.
      ordering  — neighbours are read cyclically starting from the nearest one,
                  so the sequence has a canonical origin. Without this the same
                  crater yields a different vector depending on which neighbour
                  happened to be listed first.

    An earlier version also z-scored each descriptor set against its own image
    mean and standard deviation, which quietly destroyed the whole point: two
    images with different crater populations then got different normalisations,
    so identical terrain no longer produced identical vectors.

    Returns (descriptors [N, 2k+2], craters).
    """
    n = len(craters)
    if n < k + 1:
        return np.zeros((0, 2 * k + 2), np.float32), craters
    xy = craters[:, :2].astype(np.float64)
    rad = craters[:, 2].astype(np.float64)
    tree = cKDTree(xy)
    d, idx = tree.query(xy, k=k + 1)
    d, idx = d[:, 1:], idx[:, 1:]

    descs = np.zeros((n, 2 * k + 2))
    for i in range(n):
        unit = max(np.median(d[i]), 1e-6)          # local length unit
        v = xy[idx[i]] - xy[i]
        ang = np.arctan2(v[:, 1], v[:, 0])
        order = np.argsort(ang)                    # counter-clockwise sweep
        ang, dd, nb = ang[order], d[i][order], idx[i][order]

        start = int(np.argmin(dd))                 # canonical origin
        roll = np.roll(np.arange(k), -start)
        ang, dd, nb = ang[roll], dd[roll], nb[roll]

        gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]])) / (2 * np.pi)
        descs[i] = np.concatenate([
            dd / unit,                             # k distance ratios
            gaps,                                  # k angular gaps
            [rad[i] / unit, np.median(rad[nb]) / unit],
        ])
    # fixed, data-independent weighting so both blocks contribute comparably
    w = np.concatenate([np.full(k, 1.0), np.full(k, 3.0), [1.0, 1.0]])
    return (descs * w).astype(np.float32), craters


def match_constellations(cA: np.ndarray, cB: np.ndarray, k: int = 6,
                         ratio: float = 0.88, min_inliers: int = 6,
                         thresh_frac: float = 0.02,
                         scale_range: tuple = (0.05, 20.0)):
    """
    Match two crater fields and fit a similarity transform A -> B.

    RANSAC threshold scales with image size rather than being an absolute pixel
    count, because at a 10:1 ratio a "close" match in the coarse frame is ten
    coarse pixels in the fine frame.

    Returns (H 3x3 or None, ptsA, ptsB, inlier_mask).
    """
    dA, kA = constellation_descriptors(cA, k)
    dB, kB = constellation_descriptors(cB, k)
    if len(dA) < k + 1 or len(dB) < k + 1:
        return None, np.zeros((0, 2)), np.zeros((0, 2)), None

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(dA, dB, k=2)
    pairs = [(m.queryIdx, m.trainIdx) for m in
             (p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance)]
    if len(pairs) < min_inliers:
        return None, np.zeros((0, 2)), np.zeros((0, 2)), None

    pairs = np.array(pairs, int)
    pA = kA[pairs[:, 0], :2].astype(np.float32)
    pB = kB[pairs[:, 1], :2].astype(np.float32)

    span = max(np.ptp(pB[:, 0]), np.ptp(pB[:, 1]), 1.0)
    M, mask = cv2.estimateAffinePartial2D(
        pA.reshape(-1, 1, 2), pB.reshape(-1, 1, 2), method=cv2.RANSAC,
        ransacReprojThreshold=max(2.0, thresh_frac * span),
        maxIters=8000, confidence=0.999)
    if M is None or mask is None or mask.sum() < min_inliers:
        return None, pA, pB, None
    H = np.vstack([M, [0, 0, 1.0]])
    # reject degenerate fits: RANSAC will happily collapse the transform to a
    # point if the inlier set is tiny, which reads as "success" downstream.
    sc = float(np.hypot(H[0, 0], H[0, 1]))
    if not (scale_range[0] < sc < scale_range[1]):
        return None, pA, pB, None
    return H, pA, pB, mask.ravel().astype(bool)


def crater_bootstrap(src: np.ndarray, ref: np.ndarray,
                     sun_src: float | None = None, sun_ref: float | None = None,
                     detector: str = 'matched', k: int = 4,
                     min_score: float = 0.5, max_craters: int = 200):
    """
    Metadata-free bootstrap: detect, describe, match, fit.

    k defaults to 4 rather than 6 because a constellation descriptor needs all k
    neighbours co-detected in both images. At a measured recall p the odds go as
    p^k, so with p around 0.6 dropping from 6 to 4 neighbours raises the usable
    fraction from roughly 5% to 13% -- at some cost in descriptor
    distinctiveness, which RANSAC then has to absorb.

    Only the solar azimuth is used, never the GSD, so this path stays valid when
    the scale prior is missing or wrong.
    """
    if detector == 'matched' and sun_src is not None and sun_ref is not None:
        cS = detect_craters_matched(src, sun_src, min_score=min_score,
                                    max_craters=max_craters)
        cR = detect_craters_matched(ref, sun_ref, min_score=min_score,
                                    max_craters=max_craters)
    else:
        cS = detect_craters(src, sun_azimuth_deg=sun_src)
        cR = detect_craters(ref, sun_azimuth_deg=sun_ref)
    H, pA, pB, mask = match_constellations(cS, cR, k=k)
    return dict(H=H, n_craters_src=len(cS), n_craters_ref=len(cR),
                n_putative=len(pA), n_inliers=int(mask.sum()) if mask is not None else 0,
                craters_src=cS, craters_ref=cR,
                scale=float(np.hypot(H[0, 0], H[0, 1])) if H is not None else None)


# --------------------------------------------------------------------------
# Matched-filter detection
# --------------------------------------------------------------------------
#
# Measured: Hough-on-phase-congruency gives precision 0.10-0.33 and recall
# 0.14-0.27 against the known catalogue. That is the binding constraint on the
# whole constellation approach, because a descriptor built from k neighbours
# needs all k co-detected -- at recall p the chance is roughly p^k, so p = 0.2
# with k = 6 is hopeless no matter how good the matcher is.
#
# Hough is the wrong tool here. It looks for circular EDGES, but a lunar crater
# under oblique light is not an edge ring: it is a bowl whose shading ramps
# along the illumination direction, bright on the sun-facing interior wall and
# dark opposite. Solar azimuth is in every PDS label, so that ramp is known a
# priori and can be matched directly.

def crater_template(radius: float, sun_azimuth_deg: float, rim: float = 0.35):
    """
    Appearance model of a simple bowl crater under directional illumination.

    Interior shading follows the surface normal of a paraboloid, which tilts
    linearly with radius, giving an intensity ramp along the solar azimuth. A
    raised rim adds a bright/dark annulus of opposite polarity.
    """
    r = int(np.ceil(radius * 1.35))
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    rho = np.hypot(x, y)
    a = np.deg2rad(sun_azimuth_deg)
    ux, uy = np.sin(a), -np.cos(a)
    proj = (x * ux + y * uy) / max(radius, 1e-6)

    t = np.zeros_like(rho)
    inside = rho <= radius
    # Sign matters and is easy to invert. The interior wall on the FAR side from
    # the sun has its normal tilted toward the sun and is therefore bright; the
    # near-side wall faces away and is dark. Getting this backwards turns the
    # template into a mound detector, which correlates with real terrain just
    # well enough to look like it is working while precision sits near zero.
    t[inside] = -proj[inside]
    ring = (rho > radius) & (rho <= radius * 1.35)
    t[ring] = rim * proj[ring]                     # raised rim, opposite polarity
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def detect_craters_matched(img: np.ndarray, sun_azimuth_deg: float,
                           radii=None, max_craters: int = 300,
                           min_score: float = 0.30, nms_frac: float = 0.9):
    """
    Detect craters by normalised cross-correlation against a bank of bowl
    templates at several radii.

    Returns [N, 4] of (x, y, radius_px, score). Score is the NCC peak, which is
    directly interpretable and comparable across radii.
    """
    h, w = img.shape
    if radii is None:
        rmax = max(6.0, min(h, w) * 0.13)
        radii = np.geomspace(3.0, rmax, 7)

    im = img.astype(np.float32)
    im = (im - im.mean()) / (im.std() + 1e-8)

    cand = []
    for rad in radii:
        t = crater_template(rad, sun_azimuth_deg).astype(np.float32)
        if t.shape[0] >= min(h, w):
            continue
        resp = cv2.matchTemplate(im, t, cv2.TM_CCOEFF_NORMED)
        off = t.shape[0] // 2
        # local maxima, spaced by the crater size
        ksz = int(max(3, 2 * round(rad * nms_frac) + 1))
        mx = cv2.dilate(resp, np.ones((ksz, ksz), np.uint8))
        peaks = (resp >= mx) & (resp > min_score)
        ys, xs = np.nonzero(peaks)
        for yy, xx in zip(ys, xs):
            cand.append((xx + off, yy + off, rad, float(resp[yy, xx])))

    if not cand:
        return np.zeros((0, 4), np.float32)
    cand = np.array(cand, np.float32)
    cand = cand[np.argsort(-cand[:, 3])]

    # cross-radius suppression: one detection per physical crater
    keep, taken = [], np.zeros((0, 3))
    for c in cand:
        if len(taken):
            d = np.hypot(taken[:, 0] - c[0], taken[:, 1] - c[1])
            if (d < np.maximum(taken[:, 2], c[2]) * 0.85).any():
                continue
        keep.append(c)
        taken = np.vstack([taken, c[:3][None, :]])
        if len(keep) >= max_craters:
            break
    return np.array(keep, np.float32)


# --------------------------------------------------------------------------
# Transform voting  (robust replacement for k-NN constellation descriptors)
# --------------------------------------------------------------------------
#
# Why the descriptor approach above is not enough, measured:
#   matched-filter detection reaches recall ~0.6, precision ~0.7. A k-neighbour
#   descriptor needs ALL k neighbours co-detected AND no false positive
#   intruding into the neighbourhood. Even at k=4 that leaves ~13% of
#   descriptors intact, and the spurious 30% actively rewrite neighbourhoods.
#   End to end it produced 80 putative matches and under 6 RANSAC inliers.
#
# The fix is to stop relying on any fixed neighbourhood. Every PAIR of craters
# in image A and every pair in image B together imply a scale and a rotation.
# Correct pairs all imply the SAME scale and rotation, so they pile into one bin
# of an accumulator while wrong pairs scatter. This tolerates missing craters and
# spurious ones because no individual point is load-bearing -- the same reason
# star trackers vote over triangles instead of describing neighbourhoods.

def match_by_transform_voting(cA: np.ndarray, cB: np.ndarray,
                              max_pts: int = 60, n_scale: int = 96,
                              n_rot: int = 120, scale_lim: float = 24.0,
                              min_inliers: int = 8, tol_frac: float = 0.03):
    """
    Recover a similarity A -> B by voting in (log-scale, rotation), then in
    translation. Returns (H, ptsA, ptsB, inlier_mask) or (None, ...).
    """
    if len(cA) < 4 or len(cB) < 4:
        return None, np.zeros((0, 2)), np.zeros((0, 2)), None
    A = cA[np.argsort(-cA[:, 3])][:max_pts, :2].astype(np.float64)
    B = cB[np.argsort(-cB[:, 3])][:max_pts, :2].astype(np.float64)

    def pairs(P):
        i, j = np.triu_indices(len(P), 1)
        v = P[j] - P[i]
        d = np.hypot(v[:, 0], v[:, 1])
        th = np.arctan2(v[:, 1], v[:, 0])
        ok = d > 1e-6
        return i[ok], j[ok], d[ok], th[ok]

    ia, ja, da, ta = pairs(A)
    ib, jb, db, tb = pairs(B)
    if len(da) < 3 or len(db) < 3:
        return None, np.zeros((0, 2)), np.zeros((0, 2)), None

    # (log scale, rotation) accumulator over every pair-of-pairs
    ls = np.log(db[None, :] / da[:, None])
    dr = (tb[None, :] - ta[:, None] + np.pi) % (2 * np.pi) - np.pi
    lim = np.log(scale_lim)
    m = np.abs(ls) < lim
    if m.sum() < 10:
        return None, np.zeros((0, 2)), np.zeros((0, 2)), None
    Hh, xe, ye = np.histogram2d(ls[m], dr[m], bins=[n_scale, n_rot],
                                range=[[-lim, lim], [-np.pi, np.pi]])
    # blur the accumulator so a peak split across adjacent bins still wins
    Hh = cv2.GaussianBlur(Hh.astype(np.float32), (0, 0), 1.0)
    pk = np.unravel_index(np.argmax(Hh), Hh.shape)
    s_hat = float(np.exp(0.5 * (xe[pk[0]] + xe[pk[0] + 1])))
    r_hat = float(0.5 * (ye[pk[1]] + ye[pk[1] + 1]))

    # translation accumulator, given scale and rotation
    c, sn = np.cos(r_hat), np.sin(r_hat)
    R = np.array([[c, -sn], [sn, c]]) * s_hat
    Ar = A @ R.T
    tx = B[None, :, 0] - Ar[:, None, 0]
    ty = B[None, :, 1] - Ar[:, None, 1]
    span = max(np.ptp(B[:, 0]), np.ptp(B[:, 1]), 1.0)
    tol = max(2.0, tol_frac * span)
    nb = int(np.clip(4 * span / tol, 16, 256))
    lo = min(tx.min(), ty.min()) - 1
    hi = max(tx.max(), ty.max()) + 1
    Ht, xe2, ye2 = np.histogram2d(tx.ravel(), ty.ravel(), bins=[nb, nb],
                                  range=[[lo, hi], [lo, hi]])
    Ht = cv2.GaussianBlur(Ht.astype(np.float32), (0, 0), 1.0)
    pt = np.unravel_index(np.argmax(Ht), Ht.shape)
    tx_hat = 0.5 * (xe2[pt[0]] + xe2[pt[0] + 1])
    ty_hat = 0.5 * (ye2[pt[1]] + ye2[pt[1] + 1])

    # harvest correspondences consistent with the voted transform
    pred = Ar + np.array([tx_hat, ty_hat])
    d = np.hypot(B[None, :, 0] - pred[:, None, 0], B[None, :, 1] - pred[:, None, 1])
    jbest = np.argmin(d, axis=1)
    dbest = d[np.arange(len(A)), jbest]
    sel = dbest < tol
    if sel.sum() < min_inliers:
        return None, A, B, None

    pA = A[sel].astype(np.float32)
    pB = B[jbest[sel]].astype(np.float32)
    M, mask = cv2.estimateAffinePartial2D(
        pA.reshape(-1, 1, 2), pB.reshape(-1, 1, 2), method=cv2.RANSAC,
        ransacReprojThreshold=tol * 0.8, maxIters=6000, confidence=0.999)
    if M is None or mask is None or mask.sum() < min_inliers:
        return None, pA, pB, None
    H = np.vstack([M, [0, 0, 1.0]])
    sc = float(np.hypot(H[0, 0], H[0, 1]))
    if not (1.0 / scale_lim < sc < scale_lim):
        return None, pA, pB, None
    return H, pA, pB, mask.ravel().astype(bool)


def crater_register(src, ref, sun_src, sun_ref, min_score: float = 0.45,
                    max_craters: int = 220):
    """Detection + transform voting. No GSD prior, no appearance matching."""
    cS = detect_craters_matched(src, sun_src, min_score=min_score, max_craters=max_craters)
    cR = detect_craters_matched(ref, sun_ref, min_score=min_score, max_craters=max_craters)
    H, pA, pB, mask = match_by_transform_voting(cS, cR)
    return dict(H=H, n_craters_src=len(cS), n_craters_ref=len(cR),
                n_putative=len(pA), n_inliers=int(mask.sum()) if mask is not None else 0,
                craters_src=cS, craters_ref=cR,
                scale=float(np.hypot(H[0, 0], H[0, 1])) if H is not None else None)

# ==========================================================================
# synth.py
# ==========================================================================

"""
synth.py — Physically-motivated synthetic lunar scene generator.

Why this exists
---------------
Sub-pixel RMSE cannot be validated without ground truth. Manually digitised tie
points on real OHRC/NAC pairs carry ~0.5-2 px operator noise, which is the same
order as the accuracy we are trying to prove. So we build a virtual Moon:

    DEM (crater field)  +  albedo map  ->  photometric render at (az, el)
                                       ->  sensor model (scale, MTF, noise)

Two renders of the SAME DEM under DIFFERENT sun angles and DIFFERENT sensor
models give an image pair whose true geometric mapping is a homography we
wrote down ourselves. Error is then exactly measurable.

Physics implemented
-------------------
* Crater morphology: bowl interior, raised rim, ejecta blanket, power-law
  size-frequency distribution (N(>D) ~ D^-2), plus fractal (1/f^beta) regolith.
* Cast shadows: horizon ray-march along the solar azimuth. This is the effect
  that breaks SIFT — it flips the intensity gradient across a crater when the
  sun crosses to the other side.
* Lunar-Lambert / Lommel-Seeliger photometric function with opposition surge,
  which is the standard reflectance model for airless regolith.
* Sensor chain: GSD resampling, MTF blur, photon + read noise, band-dependent
  albedo response (models the OHRC/TMC/IIRS spectral mismatch).
"""


import numpy as np
from dataclasses import dataclass, field
from scipy import ndimage


# --------------------------------------------------------------------------
# Terrain
# --------------------------------------------------------------------------

def fractal_surface(n: int, beta: float = 2.3, rng: np.random.Generator | None = None) -> np.ndarray:
    """1/f^beta fractional Brownian surface — models regolith roughness."""
    rng = rng or np.random.default_rng(0)
    fx = np.fft.fftfreq(n)[:, None]
    fy = np.fft.fftfreq(n)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, (n, n))
    surf = np.fft.ifft2(amp * np.exp(1j * phase)).real
    s = surf.std()
    return surf / s if s > 0 else surf


def _crater_profile(r: np.ndarray, D: float) -> np.ndarray:
    """
    Radial elevation profile of a simple lunar crater, normalised by diameter.

    depth/diameter ~ 0.2 for fresh simple craters (Pike 1977).
    Interior: paraboloid. Rim: raised annulus ~4% of D. Ejecta: r^-3 decay.
    """
    R = D / 2.0
    d = 0.20 * D            # depth
    h_rim = 0.040 * D       # rim height above datum
    z = np.zeros_like(r)

    inside = r < R
    # paraboloid floor rising to rim crest
    z[inside] = -d + (d + h_rim) * (r[inside] / R) ** 2

    outside = ~inside
    ro = r[outside] / R
    # ejecta blanket decaying as r^-3, continuous with rim crest
    z[outside] = h_rim * np.power(ro, -3.0)
    return z


@dataclass
class TerrainConfig:
    n: int = 768                 # DEM grid size (px)
    gsd_m: float = 1.0           # metres per DEM pixel
    n_craters: int = 220
    d_min_m: float = 6.0
    d_max_m: float = 180.0
    sfd_exponent: float = 2.0    # N(>D) ~ D^-exponent
    roughness_m: float = 0.45
    roughness_beta: float = 2.3
    albedo_mean: float = 0.11    # lunar highlands ~0.11-0.18, maria ~0.07
    albedo_contrast: float = 0.10
    ray_craters: int = 4         # fresh craters with bright ejecta rays
    seed: int = 0


def make_terrain(cfg: TerrainConfig):
    """Return (dem_metres, albedo) on an n x n grid."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n
    dem = cfg.roughness_m * fractal_surface(n, cfg.roughness_beta, rng)

    # Albedo: broad maria/highland patches + fine speckle. Albedo features are
    # illumination-invariant, shading features are not. Keeping them separable
    # lets us later attribute matcher performance to one or the other.
    alb = cfg.albedo_mean * (1.0 + cfg.albedo_contrast * fractal_surface(n, 3.0, rng))
    alb += cfg.albedo_mean * 0.03 * fractal_surface(n, 1.2, rng)

    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)

    # Power-law crater size sampling via inverse CDF
    u = rng.random(cfg.n_craters)
    a, b, k = cfg.d_min_m, cfg.d_max_m, cfg.sfd_exponent
    D = (a ** -k + u * (b ** -k - a ** -k)) ** (-1.0 / k)
    D_px = D / cfg.gsd_m
    order = np.argsort(-D_px)          # emplace large first, small overprint
    D_px = D_px[order]

    cx = rng.uniform(-0.1 * n, 1.1 * n, cfg.n_craters)
    cy = rng.uniform(-0.1 * n, 1.1 * n, cfg.n_craters)
    degrade = rng.uniform(0.25, 1.0, cfg.n_craters)   # erosion / age factor

    for i in range(cfg.n_craters):
        d = D_px[i]
        half = int(min(3.0 * d, n))
        x0, x1 = int(max(0, cx[i] - half)), int(min(n, cx[i] + half))
        y0, y1 = int(max(0, cy[i] - half)), int(min(n, cy[i] + half))
        if x1 <= x0 or y1 <= y0:
            continue
        sx = xx[y0:y1, x0:x1] - cx[i]
        sy = yy[y0:y1, x0:x1] - cy[i]
        r = np.hypot(sx, sy)
        prof = _crater_profile(r, d) * cfg.gsd_m * degrade[i]
        prof = np.clip(prof, -0.25 * d * cfg.gsd_m, 0.06 * d * cfg.gsd_m)
        dem[y0:y1, x0:x1] += prof

        # fresh small craters excavate high-albedo immature regolith
        if i >= cfg.n_craters - cfg.ray_craters and d > 6:
            halo = np.exp(-(r / (1.8 * d / 2)) ** 2)
            alb[y0:y1, x0:x1] += 0.055 * halo * degrade[i]

    alb = np.clip(alb, 0.03, 0.35)
    # Ground-truth crater catalogue in DEM pixel coordinates. Exposing this lets
    # us separate two failure modes that look identical from the outside: a
    # crater DETECTOR that misses landmarks, and a constellation MATCHER that
    # cannot use the landmarks it is given.
    cat = np.stack([cx, cy, D_px / 2.0], 1).astype(np.float64)
    return dem.astype(np.float64), alb.astype(np.float64), cat


# --------------------------------------------------------------------------
# Illumination
# --------------------------------------------------------------------------

def cast_shadow_mask(dem: np.ndarray, gsd_m: float, az_deg: float, el_deg: float,
                     max_steps: int = 96) -> np.ndarray:
    """
    Binary illumination mask by horizon ray-marching toward the sun.

    A pixel is shadowed if any terrain along the solar azimuth subtends an
    angle greater than the solar elevation. Vectorised: one shifted array per
    step, so cost is O(max_steps * n^2) but fully in numpy.
    """
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    # image convention: +x East, +y South (row index down)
    dx, dy = np.sin(az), -np.cos(az)
    tan_el = np.tan(el)

    lit = np.ones_like(dem, dtype=bool)
    for k in range(1, max_steps + 1):
        shifted = ndimage.shift(dem, shift=(-dy * k, -dx * k), order=1,
                                mode='nearest', prefilter=False)
        slope = (shifted - dem) / (k * gsd_m)
        lit &= (slope <= tan_el)
        if k > 8 and lit.all():
            break
    return lit


def render(dem: np.ndarray, albedo: np.ndarray, gsd_m: float,
           sun_az_deg: float, sun_el_deg: float,
           emission_deg: float = 0.0, shadows: bool = True,
           lunar_lambert_L: float = 0.85) -> np.ndarray:
    """
    Radiance render using the Lunar-Lambert law:

        I = A * mu0 * [ 2L * mu0/(mu0+mu) + (1-L) ]

    L=1 -> pure Lommel-Seeliger, L=0 -> Lambert. L~0.85 fits lunar regolith at
    moderate phase angles. Opposition surge added as a mild phase term.
    """
    gy, gx = np.gradient(dem, gsd_m)
    # outward surface normal
    nz = 1.0 / np.sqrt(1.0 + gx ** 2 + gy ** 2)
    nx, ny = -gx * nz, -gy * nz

    az, el = np.deg2rad(sun_az_deg), np.deg2rad(sun_el_deg)
    sx = np.cos(el) * np.sin(az)
    sy = -np.cos(el) * np.cos(az)
    sz = np.sin(el)

    mu0 = nx * sx + ny * sy + nz * sz      # cos(incidence)
    mu0 = np.clip(mu0, 0.0, None)

    e = np.deg2rad(emission_deg)
    mu = np.clip(nz * np.cos(e) + nx * np.sin(e), 1e-3, None)

    L = lunar_lambert_L
    refl = mu0 * (2.0 * L * mu0 / (mu0 + mu) + (1.0 - L))

    phase = np.abs(np.deg2rad(90.0 - sun_el_deg) - e)
    surge = 1.0 + 0.35 * np.exp(-phase / 0.12)     # opposition effect

    img = albedo * refl * surge
    if shadows:
        lit = cast_shadow_mask(dem, gsd_m, sun_az_deg, sun_el_deg)
        # shadows are not black: diffuse scattering from illuminated slopes
        img = np.where(lit, img, img * 0.06 + albedo * 0.004)
    return img


# --------------------------------------------------------------------------
# Sensor model
# --------------------------------------------------------------------------

SENSORS = {
    # gsd_m, mtf_sigma_px, snr, band_weight (spectral response proxy), bit depth
    'OHRC':  dict(gsd_m=0.32,  mtf=0.62, snr=180.0, band=1.00, bits=10),
    'TMC2':  dict(gsd_m=5.00,  mtf=0.75, snr=140.0, band=0.95, bits=10),
    'IIRS':  dict(gsd_m=80.00, mtf=0.95, snr=45.0,  band=0.55, bits=12),
    'NAC':   dict(gsd_m=0.50,  mtf=0.55, snr=200.0, band=1.00, bits=12),
    'TC':    dict(gsd_m=10.00, mtf=0.80, snr=160.0, band=0.90, bits=10),
}


@dataclass
class ViewGeometry:
    """Affine/projective viewpoint difference, in source-image pixel units."""
    scale: float = 1.0
    rotation_deg: float = 0.0
    shear: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    persp: tuple = (0.0, 0.0)   # h31, h32 — off-nadir perspective

    def matrix(self, cx: float, cy: float) -> np.ndarray:
        th = np.deg2rad(self.rotation_deg)
        c, s = np.cos(th), np.sin(th)
        A = np.array([[self.scale * c, self.scale * (-s + self.shear), 0.0],
                      [self.scale * s, self.scale * c, 0.0],
                      [0.0, 0.0, 1.0]])
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
        T2 = np.array([[1, 0, cx + self.tx], [0, 1, cy + self.ty], [0, 0, 1.0]])
        H = T2 @ A @ T1
        H[2, 0], H[2, 1] = self.persp
        return H / H[2, 2]


def apply_sensor(radiance: np.ndarray, sensor: str, dem_gsd_m: float,
                 rng: np.random.Generator, extra_blur: float = 0.0):
    """
    GSD resample -> MTF blur -> band response -> photon+read noise -> quantise.

    Returns (image, effective_zoom). The effective zoom is out_size/in_size,
    which differs from the requested factor whenever the product rounds — at a
    10:1 scale ratio that discrepancy is several reference pixels, so it must
    be propagated into the ground-truth homography rather than assumed away.
    """
    s = SENSORS[sensor]
    zoom = dem_gsd_m / s['gsd_m']
    img = radiance * s['band']
    sigma = np.hypot(s['mtf'], extra_blur)
    if zoom < 1.0:                       # downsampling: pre-filter to avoid alias
        img = ndimage.gaussian_filter(img, sigma / max(zoom, 1e-6) * 0.4)
    # grid_mode=True uses the pixel-AREA convention (out o <-> in (o+.5)/z-.5).
    # With the default grid_mode=False, zoom aligns first/last pixel CENTRES,
    # which injects a ~0.5 px scale-dependent bias — fatal when the whole point
    # of the harness is measuring sub-pixel error.
    n_in = img.shape[0]
    img = ndimage.zoom(img, zoom, order=3, mode='grid-constant', grid_mode=True)
    z_eff = img.shape[0] / n_in
    img = ndimage.gaussian_filter(img, sigma)

    m = img.mean() if img.mean() > 0 else 1.0
    img = img / m
    photons = (s['snr'] ** 2)
    img = rng.poisson(np.clip(img, 0, None) * photons) / photons
    img += rng.normal(0.0, 1.0 / (3.0 * s['snr']), img.shape)

    lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
    levels = 2 ** s['bits']
    return (np.round(img * (levels - 1)) / (levels - 1)).astype(np.float32), z_eff


@dataclass
class PairSpec:
    """Everything that defines one registration experiment."""
    src_sensor: str = 'OHRC'
    ref_sensor: str = 'NAC'
    src_sun: tuple = (135.0, 42.0)     # (azimuth, elevation) degrees
    ref_sun: tuple = (315.0, 28.0)
    view: ViewGeometry = field(default_factory=ViewGeometry)
    out_size: int = 512
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    seed: int = 0

    @property
    def sun_delta(self) -> dict:
        d_az = abs(self.src_sun[0] - self.ref_sun[0]) % 360.0
        d_az = min(d_az, 360.0 - d_az)
        return dict(d_az=d_az, d_el=abs(self.src_sun[1] - self.ref_sun[1]))


def auto_spec(src_sensor: str, ref_sensor: str, *,
              src_sun=(135.0, 42.0), ref_sun=(315.0, 28.0),
              view: ViewGeometry | None = None,
              coarse_px: int = 384, max_px: int = 720, seed: int = 0) -> PairSpec:
    """
    Build a PairSpec whose DEM actually spans the ground both sensors need.

    A 10:1 GSD ratio means the coarse image covers 10x the ground per pixel, so
    a DEM fine enough for the sharp sensor and wide enough for the coarse one
    explodes in size. We therefore size the footprint from the COARSE sensor
    and cap the fine image at max_px — which is also the physical reality of
    such a pairing: you register a small coarse chip against a fine strip.
    """
    gs, gr = SENSORS[src_sensor]['gsd_m'], SENSORS[ref_sensor]['gsd_m']
    ratio = max(gs, gr) / min(gs, gr)
    coarse_px = int(np.clip(min(coarse_px, max_px / ratio), 96, max_px))
    footprint_m = coarse_px * max(gs, gr)

    dem_gsd = min(gs, gr)
    n_dem = int(round(footprint_m / dem_gsd * 1.30))     # 30% margin for warps
    n_dem = int(np.clip(n_dem, 256, 1600))

    # crater population scaled to the footprint, not to pixel counts
    d_min = max(5.0 * dem_gsd, footprint_m / 220.0)
    d_max = footprint_m / 3.5
    area_km2 = (footprint_m * 1.3 / 1000.0) ** 2
    n_craters = int(np.clip(190 * area_km2 / max(area_km2, 1e-9), 120, 320))
    n_craters = int(np.clip(220 * (n_dem / 700.0) ** 1.1, 120, 340))

    tc = TerrainConfig(n=n_dem, gsd_m=dem_gsd, n_craters=n_craters,
                       d_min_m=d_min, d_max_m=d_max,
                       roughness_m=max(0.35 * dem_gsd, 0.05), seed=seed)
    # out_size is expressed in REFERENCE pixels
    out_size = int(round(footprint_m / gr))
    out_size = int(np.clip(out_size, 96, max_px))
    return PairSpec(src_sensor=src_sensor, ref_sensor=ref_sensor,
                    src_sun=src_sun, ref_sun=ref_sun,
                    view=view or ViewGeometry(), out_size=out_size,
                    terrain=tc, seed=seed)


def make_pair(spec: PairSpec):
    """
    Build (source, reference, H_true, meta).

    H_true maps SOURCE pixel coords -> REFERENCE pixel coords, exactly.
    """
    rng = np.random.default_rng(spec.seed + 9973)
    dem, alb, cat = make_terrain(spec.terrain)
    g = spec.terrain.gsd_m

    rad_src = render(dem, alb, g, *spec.src_sun, emission_deg=0.0)
    rad_ref = render(dem, alb, g, *spec.ref_sun, emission_deg=6.0)

    src_full, z_src = apply_sensor(rad_src, spec.src_sensor, g, rng)
    ref_full, z_ref = apply_sensor(rad_ref, spec.ref_sensor, g, rng)

    N = spec.out_size
    # crop both to a common centred window in *DEM* space, then account for the
    # differing sensor GSDs analytically so H_true stays exact.
    def centre_crop(im, N):
        h, w = im.shape
        N = min(N, h, w)
        y0, x0 = (h - N) // 2, (w - N) // 2
        return im[y0:y0 + N, x0:x0 + N], (x0, y0)

    ref_img, (rx0, ry0) = centre_crop(ref_full, N)
    # true ratio from what the resampler actually did, not from nominal GSDs
    scale_ratio = z_src / z_ref

    # Source is rendered at its own GSD then warped by the viewpoint transform.
    Nsrc = int(round(N * scale_ratio))
    src_crop, (sx0, sy0) = centre_crop(src_full, Nsrc)
    Ns = src_crop.shape[0]

    # Map src_crop pixel -> ref_img pixel. Derived, not assumed:
    #   full-frame src pixel u and ref pixel v view the same DEM point when
    #   (u+0.5)/z_src = (v+0.5)/z_ref, i.e. v = k(u+0.5) - 0.5 with k = z_ref/z_src.
    #   Substituting the crop origins u = p_s + s0, v = p_r + r0 gives the offset.
    k = 1.0 / scale_ratio
    off_x = k * (sx0 + 0.5) - 0.5 - rx0
    off_y = k * (sy0 + 0.5) - 0.5 - ry0
    H_base = np.array([[k, 0, off_x],
                       [0, k, off_y],
                       [0, 0, 1.0]])

    V = spec.view.matrix(Ns / 2.0, Ns / 2.0)
    Vi = np.linalg.inv(V)
    src_img = _warp(src_crop, Vi, (Ns, Ns))
    # _warp(img, M) maps input -> output by M. Warping by Vi therefore means a
    # pixel p of src_img corresponds to pixel V @ p of src_crop, so the source
    # -> reference chain composes with V, not Vi. (Composing Vi here is silent
    # under an identity view, which is exactly why it survives a naive test.)
    H_true = H_base @ V
    H_true = H_true / H_true[2, 2]

    # project the true crater catalogue into both image frames
    zr = z_ref
    cr = np.stack([zr * (cat[:, 0] + 0.5) - 0.5 - rx0,
                   zr * (cat[:, 1] + 0.5) - 0.5 - ry0,
                   cat[:, 2] * zr], 1)
    Hi = np.linalg.inv(H_true)
    P = np.hstack([cr[:, :2], np.ones((len(cr), 1))]).T
    Q = Hi @ P
    cs_xy = (Q[:2] / Q[2]).T
    src_scale = 1.0 / max(np.hypot(H_true[0, 0], H_true[1, 0]), 1e-9)
    cs = np.column_stack([cs_xy, cr[:, 2] * src_scale])
    inR = ((cr[:, 0] > 0) & (cr[:, 1] > 0) &
           (cr[:, 0] < ref_img.shape[1]) & (cr[:, 1] < ref_img.shape[0]))
    inS = ((cs[:, 0] > 0) & (cs[:, 1] > 0) &
           (cs[:, 0] < src_img.shape[1]) & (cs[:, 1] < src_img.shape[0]))
    keep = inR & inS

    meta = dict(scale_ratio=float(scale_ratio),
                src_shape=src_img.shape, ref_shape=ref_img.shape,
                craters_ref=cr[keep], craters_src=cs[keep],
                **spec.sun_delta)
    return src_img.astype(np.float32), ref_img.astype(np.float32), H_true, meta


def _warp(img: np.ndarray, H: np.ndarray, out_shape) -> np.ndarray:
    """Backward-map warp: out(p) = img(H^-1 p). Uses cubic interpolation."""
    h, w = out_shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ones = np.ones_like(xx)
    P = np.stack([xx, yy, ones], 0).reshape(3, -1)
    Hinv = np.linalg.inv(H)
    Q = Hinv @ P
    Q = Q[:2] / Q[2]
    out = ndimage.map_coordinates(img, [Q[1].reshape(h, w), Q[0].reshape(h, w)],
                                  order=3, mode='nearest')
    return out

# ==========================================================================
# io.py
# ==========================================================================

"""
io.py — loading real mission products (OHRC / TMC-2 / IIRS / LRO NAC / SELENE)
into the (image, metadata) shape the rest of lunareg expects.

Everything upstream of this module (pipeline.register, dense.*, features.*)
only ever sees a 2-D float array plus, optionally, a scale prior. This module
is the only place that has to know about mission file formats, so it is where
new formats get added.

Supported inputs
-----------------
PDS3   Attached-or-detached ODL label + raw binary image. This is the format
       LRO NAC EDR/CDR products use. Parser is a small, dependency-free ODL
       reader — good enough for the keys registration actually needs
       (dimensions, sample type/bits, byte order, image pointer, and whatever
       illumination/scale keywords are present), not a full ODL implementation.
PDS4   XML label + separate raw array file (.img/.qub/.dat). This is the
       format Chandrayaan-2 products (OHRC, TMC-2, IIRS) are archived in on
       ISSDC/PRADAN. ISRO's PDS4 dictionaries vary by instrument, so the
       reader is deliberately schema-tolerant: it locates the data file and
       array geometry through the standard PDS4 core classes
       (File_Area_Observational / Array_2D_Image / Array_3D_Spectrum) and
       scans for illumination/scale keywords by local tag name rather than a
       fixed namespace+path, then reports which of them it actually found.
GeoTIFF/TIFF  Already map-projected reference products (LRO NAC mosaics from
       QuickMap, SELENE strips). Uses rasterio when installed (recommended —
       it also recovers the pixel size from the affine transform); otherwise
       falls back to a plain raster read with no geometric metadata.

Every reader returns a `LunarImage`: the 2-D array (band-averaged if the
product is multi-band, e.g. IIRS) plus a `LunarImageMeta` populated with
whatever the source actually contained. Fields it could not determine are
left `None` — callers (see cli.py) fall back to a user-supplied value or to
`use_scale_prior=False` rather than silently assuming something.
"""


import os
import re
import struct
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

import numpy as np


# --------------------------------------------------------------------------
# Public result type
# --------------------------------------------------------------------------

@dataclass
class LunarImageMeta:
    path: str
    format: str                              # 'pds3' | 'pds4' | 'geotiff' | 'raster'
    lines: int
    samples: int
    bands: int = 1
    gsd_m: float | None = None               # map/ground sample distance, metres/pixel
    sun_azimuth_deg: float | None = None
    sun_elevation_deg: float | None = None
    incidence_angle_deg: float | None = None
    emission_angle_deg: float | None = None
    instrument: str | None = None
    notes: list = field(default_factory=list)


@dataclass
class LunarImage:
    data: np.ndarray                         # float32, 2-D, band-averaged if needed
    meta: LunarImageMeta


def scale_prior(src_meta: LunarImageMeta, ref_meta: LunarImageMeta) -> float | None:
    """ref_gsd / src_gsd, matching what pipeline.register expects. None if unknown."""
    if src_meta.gsd_m is None or ref_meta.gsd_m is None:
        return None
    return float(ref_meta.gsd_m / src_meta.gsd_m)


def sun_azimuth_delta(src_meta: LunarImageMeta, ref_meta: LunarImageMeta) -> float | None:
    """Signed difference in degrees for PipelineConfig.d_azimuth_prior. None if unknown."""
    if src_meta.sun_azimuth_deg is None or ref_meta.sun_azimuth_deg is None:
        return None
    d = (ref_meta.sun_azimuth_deg - src_meta.sun_azimuth_deg) % 360.0
    return float(d)


# --------------------------------------------------------------------------
# PDS3 (ODL) — LRO NAC and similar attached/detached-label products
# --------------------------------------------------------------------------

_PDS3_SAMPLE_TYPES = {
    # (SAMPLE_TYPE, SAMPLE_BITS) -> numpy dtype
    ('MSB_INTEGER', 8): '>i1', ('LSB_INTEGER', 8): '<i1',
    ('MSB_UNSIGNED_INTEGER', 8): '>u1', ('LSB_UNSIGNED_INTEGER', 8): '<u1',
    ('MSB_INTEGER', 16): '>i2', ('LSB_INTEGER', 16): '<i2',
    ('MSB_UNSIGNED_INTEGER', 16): '>u2', ('LSB_UNSIGNED_INTEGER', 16): '<u2',
    ('MSB_INTEGER', 32): '>i4', ('LSB_INTEGER', 32): '<i4',
    ('MSB_UNSIGNED_INTEGER', 32): '>u4', ('LSB_UNSIGNED_INTEGER', 32): '<u4',
    ('IEEE_REAL', 32): '>f4', ('PC_REAL', 32): '<f4',
    ('IEEE_REAL', 64): '>f8', ('PC_REAL', 64): '<f8',
}

# keys we scan for, tried in this order, across both PDS3 and PDS4 labels
_AZ_KEYS = ('SUB_SOLAR_AZIMUTH', 'SOLAR_AZIMUTH', 'sun_azimuth', 'solar_azimuth',
            'incidence_azimuth')
_EL_KEYS = ('SUB_SOLAR_ELEVATION', 'SOLAR_ELEVATION', 'sun_elevation', 'solar_elevation')
_INC_KEYS = ('INCIDENCE_ANGLE', 'incidence_angle')
_EMI_KEYS = ('EMISSION_ANGLE', 'emission_angle')
_GSD_KEYS = ('MAP_SCALE', 'MAP_RESOLUTION', 'PIXEL_ASPECT_RATIO')  # metres/px or px/degree


def _parse_pds3_label(text: str) -> dict:
    """
    Minimal ODL parser: KEY = VALUE pairs and nested OBJECT/END_OBJECT blocks.

    Returns a flat dict of the top-level keys plus one dict per OBJECT block
    (keyed by 'OBJECT:<name>'), which is all the geometry/pointer keys
    registration needs — this is not a general ODL/PVL implementation.
    """
    out: dict = {}
    stack = [out]
    names = []
    for raw in text.splitlines():
        line = raw.split('/*')[0].strip()
        if not line or line == 'END':
            continue
        m = re.match(r'^(OBJECT|GROUP)\s*=\s*(.+)$', line, re.I)
        if m:
            name = m.group(2).strip().strip('"')
            names.append(name)
            blk: dict = {}
            stack[-1][f'{m.group(1).upper()}:{name}'] = blk
            stack.append(blk)
            continue
        m = re.match(r'^(END_OBJECT|END_GROUP)\b', line, re.I)
        if m:
            if len(stack) > 1:
                stack.pop()
                names.pop()
            continue
        m = re.match(r'^(\^?[A-Za-z0-9_:]+)\s*=\s*(.+)$', line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            stack[-1][key] = _pds3_value(val)
    return out


def _pds3_value(val: str):
    val = val.strip()
    m = re.match(r'^\(?\s*([\-0-9.eE]+)\s*<([A-Za-z/]+)>\s*\)?$', val)
    if m:
        return float(m.group(1))
    val = val.strip('"')
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _find_key(label: dict, keys) -> float | None:
    """Depth-first search for any of `keys` (case-insensitive) anywhere in the label."""
    for k in keys:
        kl = k.lower()
        stack = [label]
        while stack:
            d = stack.pop()
            for kk, vv in d.items():
                base = kk.split(':', 1)[-1].lower()
                if base == kl and isinstance(vv, (int, float)):
                    return float(vv)
                if isinstance(vv, dict):
                    stack.append(vv)
    return None


def read_pds3(path: str) -> LunarImage:
    """Read a PDS3 product. `path` is the .IMG (attached label) or .LBL file."""
    with open(path, 'rb') as f:
        raw = f.read()
    # attached labels end at an explicit END line before the binary image object
    text = raw.decode('latin-1', errors='replace')
    end = re.search(r'^\s*END\s*$', text, re.M)
    header_text = text[:end.end()] if end else text
    label = _parse_pds3_label(header_text)

    record_bytes = int(label.get('RECORD_BYTES', 0) or 0)
    img_obj = label.get('OBJECT:IMAGE')
    if img_obj is None:
        raise ValueError(f'{path}: no IMAGE object found in PDS3 label')

    lines = int(img_obj['LINES'])
    samples = int(img_obj['LINE_SAMPLES'])
    bands = int(img_obj.get('BANDS', 1))
    sample_bits = int(img_obj['SAMPLE_BITS'])
    sample_type = str(img_obj['SAMPLE_TYPE']).upper()
    dtype = _PDS3_SAMPLE_TYPES.get((sample_type, sample_bits))
    if dtype is None:
        raise ValueError(f'{path}: unsupported SAMPLE_TYPE/BITS '
                         f'{sample_type}/{sample_bits}')

    image_ptr = label.get('^IMAGE', 1)
    if isinstance(image_ptr, str):
        # detached label pointing at another file: "(\"FOO.IMG\", 1)" or a bare filename
        m = re.search(r'"?([^",()]+\.[A-Za-z0-9]+)"?\s*(?:,\s*(\d+))?', image_ptr)
        data_path = os.path.join(os.path.dirname(path), m.group(1)) if m else path
        start_record = int(m.group(2)) if (m and m.group(2)) else 1
        with open(data_path, 'rb') as f:
            raw = f.read()
    else:
        start_record = int(image_ptr)

    offset = (start_record - 1) * record_bytes if record_bytes else 0
    itemsize = sample_bits // 8
    n = lines * samples * bands
    buf = raw[offset:offset + n * itemsize]
    if len(buf) < n * itemsize:
        raise ValueError(f'{path}: truncated image data '
                         f'({len(buf)} bytes, expected {n * itemsize})')
    arr = np.frombuffer(buf, dtype=np.dtype(dtype), count=n)
    arr = arr.reshape(bands, lines, samples) if bands > 1 else arr.reshape(lines, samples)
    data = arr.astype(np.float32) if bands == 1 else arr.astype(np.float32).mean(axis=0)

    meta = LunarImageMeta(
        path=path, format='pds3', lines=lines, samples=samples, bands=bands,
        gsd_m=_pds3_gsd(label), sun_azimuth_deg=_find_key(label, _AZ_KEYS),
        sun_elevation_deg=_find_key(label, _EL_KEYS),
        incidence_angle_deg=_find_key(label, _INC_KEYS),
        emission_angle_deg=_find_key(label, _EMI_KEYS),
        instrument=_pds3_str(label, ('INSTRUMENT_NAME', 'INSTRUMENT_ID')))
    return LunarImage(data, meta)


def _pds3_str(label: dict, keys) -> str | None:
    for k in keys:
        v = label.get(k)
        if isinstance(v, str):
            return v
    return None


def _pds3_gsd(label: dict) -> float | None:
    """MAP_SCALE is conventionally km/pixel in PDS3 map-projection labels."""
    mp = label.get('OBJECT:IMAGE_MAP_PROJECTION')
    if isinstance(mp, dict) and 'MAP_SCALE' in mp:
        try:
            return float(mp['MAP_SCALE']) * 1000.0
        except (TypeError, ValueError):
            pass
    return None


# --------------------------------------------------------------------------
# PDS4 (XML label + separate array file) — Chandrayaan-2 OHRC/TMC-2/IIRS
# --------------------------------------------------------------------------

_PDS4_DTYPES = {
    'IEEE754MSBSingle': '>f4', 'IEEE754LSBSingle': '<f4',
    'IEEE754MSBDouble': '>f8', 'IEEE754LSBDouble': '<f8',
    'SignedMSB2': '>i2', 'SignedLSB2': '<i2',
    'UnsignedMSB2': '>u2', 'UnsignedLSB2': '<u2',
    'SignedMSB4': '>i4', 'SignedLSB4': '<i4',
    'UnsignedMSB4': '>u4', 'UnsignedLSB4': '<u4',
    'UnsignedByte': 'u1', 'SignedByte': 'i1',
}


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def _xml_find_local(root: ET.Element, name: str):
    for el in root.iter():
        if _local(el.tag) == name:
            return el
    return None


def _xml_find_all_local(root: ET.Element, name: str):
    return [el for el in root.iter() if _local(el.tag) == name]


def _xml_scan_float(root: ET.Element, keys) -> float | None:
    keys_l = {k.lower() for k in keys}
    for el in root.iter():
        if _local(el.tag).lower() in keys_l and el.text:
            try:
                return float(el.text.strip())
            except ValueError:
                continue
    return None


def read_pds4(label_path: str) -> LunarImage:
    """
    Read a PDS4-labelled product: XML label + a separate Array_2D_Image (or
    Array_3D_Spectrum, e.g. IIRS hyperspectral cubes, averaged over bands).
    """
    root = ET.parse(label_path).getroot()
    file_area = _xml_find_local(root, 'File_Area_Observational')
    if file_area is None:
        raise ValueError(f'{label_path}: no File_Area_Observational in PDS4 label')

    file_el = _xml_find_local(file_area, 'File')
    file_name = _xml_find_local(file_el, 'file_name').text.strip()
    data_path = os.path.join(os.path.dirname(label_path), file_name)

    array_el = _xml_find_local(file_area, 'Array_2D_Image')
    is_cube = False
    if array_el is None:
        array_el = _xml_find_local(file_area, 'Array_3D_Spectrum')
        is_cube = True
    if array_el is None:
        array_el = _xml_find_local(file_area, 'Array_3D_Image')
        is_cube = array_el is not None
    if array_el is None:
        raise ValueError(f'{label_path}: no Array_2D_Image/Array_3D_* element found')

    offset_el = _xml_find_local(array_el, 'offset')
    offset = int(offset_el.text) if offset_el is not None else 0

    dtype_el = _xml_find_local(array_el, 'data_type')
    dtype = _PDS4_DTYPES.get(dtype_el.text.strip()) if dtype_el is not None else None
    if dtype is None:
        raise ValueError(f'{label_path}: unsupported/unknown data_type '
                         f"'{dtype_el.text if dtype_el is not None else None}'")

    axes = {}
    for ax in _xml_find_all_local(array_el, 'Axis_Array'):
        seq = int(_xml_find_local(ax, 'sequence_number').text)
        elements = int(_xml_find_local(ax, 'elements').text)
        name = _xml_find_local(ax, 'axis_name')
        axes[seq] = (name.text.strip().lower() if name is not None else '', elements)

    dims = [axes[k][1] for k in sorted(axes)]
    dim_names = [axes[k][0] for k in sorted(axes)]

    with open(data_path, 'rb') as f:
        f.seek(offset)
        n = int(np.prod(dims))
        buf = f.read(n * np.dtype(dtype).itemsize)
    arr = np.frombuffer(buf, dtype=np.dtype(dtype), count=n).reshape(dims)

    if is_cube:
        band_axis = next((i for i, nm in enumerate(dim_names)
                          if 'band' in nm or 'spectral' in nm), 0)
        data = arr.astype(np.float32).mean(axis=band_axis)
        bands = dims[band_axis]
        lines, samples = [d for i, d in enumerate(dims) if i != band_axis]
    else:
        data = arr.astype(np.float32)
        bands = 1
        lines, samples = dims[-2], dims[-1]

    notes = []
    gsd = _xml_scan_float(root, ('pixel_resolution_x', 'spatial_resolution',
                                 'pixel_size', 'ground_sample_distance'))
    if gsd is None:
        notes.append('no GSD keyword found in PDS4 label; pass --src-gsd/--ref-gsd '
                     'explicitly or disable the scale prior')
    az = _xml_scan_float(root, _AZ_KEYS)
    el = _xml_scan_float(root, _EL_KEYS)
    if az is None or el is None:
        notes.append('no solar azimuth/elevation found; d_azimuth_prior will be unset')

    meta = LunarImageMeta(
        path=label_path, format='pds4', lines=int(lines), samples=int(samples),
        bands=int(bands), gsd_m=gsd, sun_azimuth_deg=az, sun_elevation_deg=el,
        incidence_angle_deg=_xml_scan_float(root, _INC_KEYS),
        emission_angle_deg=_xml_scan_float(root, _EMI_KEYS),
        instrument=(lambda e: e.text.strip() if e is not None else None)(
            _xml_find_local(root, 'instrument_name')),
        notes=notes)
    return LunarImage(data, meta)


# --------------------------------------------------------------------------
# GeoTIFF / plain raster — map-projected reference products
# --------------------------------------------------------------------------

def read_raster(path: str) -> LunarImage:
    """
    Georeferenced raster (rasterio, when installed — recovers true GSD from
    the affine transform) or a plain image otherwise (cv2 fallback, no GSD).
    """
    try:
        import rasterio
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float32)
            t = ds.transform
            gsd = float((abs(t.a) + abs(t.e)) / 2.0)
            if ds.crs is not None and ds.crs.is_geographic:
                # transform is in degrees/pixel; convert using the mean lunar radius
                gsd = float(np.deg2rad(gsd) * 1_737_400.0)
            meta = LunarImageMeta(path=path, format='geotiff', lines=arr.shape[0],
                                  samples=arr.shape[1], gsd_m=gsd)
            return LunarImage(arr, meta)
    except ImportError:
        pass

    import cv2
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f'{path}: could not be read as an image')
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    meta = LunarImageMeta(path=path, format='raster', lines=arr.shape[0],
                          samples=arr.shape[1],
                          notes=['rasterio not installed: no GSD/geo metadata read'])
    return LunarImage(arr.astype(np.float32), meta)


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------

def load_image(path: str) -> LunarImage:
    """Detect the product format from extension/content and load it."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xml',):
        return read_pds4(path)
    if ext in ('.lbl',):
        with open(path, 'rb') as f:
            head = f.read(2048)
        if head.lstrip().startswith(b'<?xml') or b'<Product_Observational' in head:
            return read_pds4(path)
        return read_pds3(path)
    if ext in ('.img', '.qub', '.dat'):
        sibling_xml = os.path.splitext(path)[0] + '.xml'
        if os.path.exists(sibling_xml):
            return read_pds4(sibling_xml)
        with open(path, 'rb') as f:
            head = f.read(2048)
        if head.lstrip().upper().startswith(b'PDS_VERSION_ID') or b'ODL_VERSION_ID' in head:
            return read_pds3(path)
        raise ValueError(f'{path}: cannot determine label format '
                         '(no matching .xml PDS4 label, no PDS3 header)')
    if ext in ('.tif', '.tiff'):
        return read_raster(path)
    return read_raster(path)

# ==========================================================================
# pipeline.py
# ==========================================================================

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


import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy import ndimage


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
            out['rmse_coarse'] = true_geometric_error(
                H0 @ H_pre, H_true, src.shape)['rmse_true']

        t = time.time()
        H_d, ps, pr, q = dense_register(src_w, ref, H0)
        out['t_dense'] = time.time() - t
        out['n_putative'] = int(len(ps))
        if len(ps) < 8:
            out['fail'] = 'dense matching failed'
            out['t_total'] = time.time() - t0
            return out

        out['uniformity_before'] = uniformity_report(pr, ref.shape, cfg.uniform_grid)
        if cfg.uniform and len(ps) > cfg.target_points // 2:
            keep = enforce_uniform(ps, pr, q, ref.shape,
                                        target=cfg.target_points, grid=cfg.uniform_grid)
            if len(keep) >= 12:
                ps, pr, q = ps[keep], pr[keep], q[keep]
        out['uniformity_after'] = uniformity_report(pr, ref.shape, cfg.uniform_grid)

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

        res = residuals(H_fin, ps, pr)
        r_, lo, hi = rmse_bootstrap_ci(res)
        out['rmse_reproj'] = r_
        out['rmse_ci'] = (lo, hi)
        out.update({f'resid_{k}': v for k, v in residual_diagnostics(pr, res).items()})
        if H_true is not None:
            out.update(true_geometric_error(out['H'], H_true, src.shape))
            origA = apply_H(np.linalg.inv(H_pre), ps)
            out.update(match_quality(origA, pr, H_true, tol=3.0))
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
    out['uniformity_before'] = uniformity_report(inB, shape_ref, cfg.uniform_grid)
    if cfg.uniform and len(inA) > cfg.target_points // 2:
        keep = enforce_uniform(inA, inB, qual, shape_ref,
                                    target=cfg.target_points, grid=cfg.uniform_grid)
        if len(keep) >= 12:
            inA, inB = inA[keep], inB[keep]
    out['uniformity_after'] = uniformity_report(inB, shape_ref, cfg.uniform_grid)

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
    res = residuals(H_fin, inA, inB)
    r, lo, hi = rmse_bootstrap_ci(res)
    out['rmse_reproj'] = r
    out['rmse_ci'] = (lo, hi)
    out.update({f'resid_{k}': v for k, v in residual_diagnostics(inB, res).items()})

    if H_true is not None:
        out.update(true_geometric_error(H_final, H_true, src.shape))
        # correct-match statistics need points in ORIGINAL source coordinates
        Hpi = np.linalg.inv(H_pre)
        origA = apply_H(Hpi, inA)
        out.update(match_quality(origA, inB, H_true, tol=3.0))

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

# ==========================================================================
# validate.py
# ==========================================================================

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


import numpy as np


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

# ==========================================================================
# experiments.py
# ==========================================================================

"""
experiments.py — the statistical study.

Five experiments, all against exact ground truth:

  E1 illumination   method x solar azimuth difference x seed
  E2 scale          cross-sensor pairings spanning 1.6:1 to 16:1 GSD ratios
  E3 ablation       switch off one pipeline stage at a time
  E4 noise          sensor SNR degradation
  E5 uniformity     spatial distribution of tie points, sparse vs dense

Each run records true geometric error (against the known homography), the
self-consistency metrics you could report on real data, residual structure, and
timings. Results go to CSV so the analysis is reproducible and auditable.
"""


import itertools
import sys
import time

import numpy as np
import pandas as pd


DEFAULT_VIEW = dict(rotation_deg=6.0, tx=6.0, ty=-4.0)


def _row(tag, meth, o, extra):
    r = dict(experiment=tag, method=meth, ok=bool(o.get('ok', False)))
    for k in ('rmse_true', 'max_true', 'bias_x', 'bias_y', 'rmse_reproj',
              'rmse_coarse', 'inlier_ratio', 'n_final', 'n_putative', 'ncm',
              'cmr', 'n_inliers_raw', 't_total', 'model_used', 'mim_shift',
              'resid_moran_x', 'resid_moran_y', 'resid_bias', 'resid_aniso'):
        r[k] = o.get(k, np.nan)
    for phase in ('before', 'after'):
        u = o.get(f'uniformity_{phase}')
        if isinstance(u, dict):
            for k, v in u.items():
                r[f'unif_{phase}_{k}'] = v
    lo_hi = o.get('rmse_ci', (np.nan, np.nan))
    r['rmse_ci_lo'], r['rmse_ci_hi'] = lo_hi[0], lo_hi[1]
    r['fail'] = o.get('fail', '')
    r.update(extra)
    return r


def _run(src, ref, H, m, meth, cfg_kw=None, extra=None, tag=''):
    cfg = PipelineConfig(method=meth, **(cfg_kw or {}))
    prior = m['scale_ratio'] if cfg.use_scale_prior else None
    try:
        o = register(src, ref, cfg, scale_prior=prior, H_true=H)
    except Exception as e:                                    # noqa: BLE001
        o = dict(ok=False, fail=f'exception: {type(e).__name__}')
    return _row(tag, meth, o, dict(extra or {}))


# --------------------------------------------------------------------------

def e1_illumination(seeds=(1, 2, 3, 4, 5), azimuths=(0, 30, 60, 90, 120, 150, 180),
                    methods=('sift', 'akaze', 'pcsift', 'rift', 'dense'),
                    d_elev=(35.0, 22.0)):
    rows = []
    for seed, daz in itertools.product(seeds, azimuths):
        sp = auto_spec('OHRC', 'NAC', src_sun=(135.0, d_elev[0]),
                       ref_sun=(135.0 + daz, d_elev[1]),
                       view=ViewGeometry(**DEFAULT_VIEW), seed=seed)
        src, ref, H, m = make_pair(sp)
        for meth in methods:
            rows.append(_run(src, ref, H, m, meth,
                             extra=dict(d_az=daz, seed=seed,
                                        d_el=abs(d_elev[0] - d_elev[1])),
                             tag='E1_illumination'))
        print(f'  E1 seed={seed} daz={daz} done', flush=True)
    return pd.DataFrame(rows)


def e2_scale(seeds=(1, 2, 3),
             pairs=(('OHRC', 'NAC'), ('NAC', 'OHRC'), ('TMC2', 'TC'),
                    ('IIRS', 'TC'), ('OHRC', 'TMC2'), ('TMC2', 'NAC')),
             methods=('sift', 'pcsift', 'rift', 'dense'), daz=90.0):
    rows = []
    for seed, (a, b) in itertools.product(seeds, pairs):
        sp = auto_spec(a, b, src_sun=(135.0, 35.0), ref_sun=(135.0 + daz, 30.0),
                       view=ViewGeometry(**DEFAULT_VIEW), seed=seed)
        src, ref, H, m = make_pair(sp)
        ratio = SENSORS[a]['gsd_m'] / SENSORS[b]['gsd_m']
        for meth in methods:
            rows.append(_run(src, ref, H, m, meth,
                             extra=dict(src_sensor=a, ref_sensor=b, seed=seed,
                                        gsd_ratio=ratio,
                                        scale_ratio=m['scale_ratio'], d_az=daz),
                             tag='E2_scale'))
        print(f'  E2 seed={seed} {a}->{b} done', flush=True)
    return pd.DataFrame(rows)


def e3_ablation(seeds=(1, 2, 3, 4), azimuths=(0, 90, 180)):
    variants = {
        'full':              dict(),
        'no_phase_congruency': dict(),   # handled below via dense_use_pc
        'no_subpixel':       dict(subpixel=False),
        'no_uniformity':     dict(uniform=False),
        'no_scale_prior':    dict(use_scale_prior=False),
    }
    rows = []
    for seed, daz in itertools.product(seeds, azimuths):
        sp = auto_spec('OHRC', 'NAC', src_sun=(135.0, 35.0),
                       ref_sun=(135.0 + daz, 22.0),
                       view=ViewGeometry(**DEFAULT_VIEW), seed=seed)
        src, ref, H, m = make_pair(sp)
        for name, kw in variants.items():
            if name == 'no_phase_congruency':
                rows.append(_ablate_no_pc(src, ref, H, m, seed, daz))
                continue
            rows.append(_run(src, ref, H, m, 'dense', cfg_kw=kw,
                             extra=dict(variant=name, seed=seed, d_az=daz),
                             tag='E3_ablation'))
        print(f'  E3 seed={seed} daz={daz} done', flush=True)
    return pd.DataFrame(rows)


def _ablate_no_pc(src, ref, H, m, seed, daz):
    """Dense pipeline with raw intensity instead of phase congruency."""

    t0 = time.time()
    o = dict(ok=False)
    try:
        src_w, H_pre = _prescale(src, 1.0 / m['scale_ratio'])
        H0 = coarse_align(src_w, ref, )
        Hd = H0
        ps = pr = np.zeros((0, 2))
        for (g, w, s) in ((10, 64, 24), (16, 40, 8), (22, 32, 4)):
            ps, pr, q = grid_tiepoints(src_w, ref, Hd, grid=g, win=w,
                                         search=s, use_pc=False)
            if len(ps) < 8:
                break
            Hn, mask, _ = estimate_progressive(ps, pr, 2.0)
            if Hn is None:
                break
            Hd, ps, pr = Hn, ps[mask], pr[mask]
        if len(ps) >= 8:
            Hf = Hd @ H_pre
            o = dict(ok=True, n_final=len(ps), n_putative=len(ps))
            o.update(true_geometric_error(Hf / Hf[2, 2], H, src.shape))
        else:
            o['fail'] = 'too few tie points'
    except Exception as e:                                    # noqa: BLE001
        o['fail'] = f'exception: {type(e).__name__}'
    o['t_total'] = time.time() - t0
    return _row('E3_ablation', 'dense', o,
                dict(variant='no_phase_congruency', seed=seed, d_az=daz))


def e4_noise(seeds=(1, 2, 3), snr_scales=(1.0, 0.5, 0.25, 0.12),
             methods=('pcsift', 'rift', 'dense'), daz=90.0):
    rows = []
    base = SENSORS['OHRC']['snr'], SENSORS['NAC']['snr']
    for seed, k in itertools.product(seeds, snr_scales):
        SENSORS['OHRC']['snr'] = base[0] * k
        SENSORS['NAC']['snr'] = base[1] * k
        try:
            sp = auto_spec('OHRC', 'NAC', src_sun=(135.0, 35.0),
                           ref_sun=(135.0 + daz, 30.0),
                           view=ViewGeometry(**DEFAULT_VIEW), seed=seed)
            src, ref, H, m = make_pair(sp)
            for meth in methods:
                rows.append(_run(src, ref, H, m, meth,
                                 extra=dict(snr_scale=k, seed=seed,
                                            snr_src=SENSORS['OHRC']['snr']),
                                 tag='E4_noise'))
        finally:
            SENSORS['OHRC']['snr'], SENSORS['NAC']['snr'] = base
        print(f'  E4 seed={seed} snr_scale={k} done', flush=True)
    return pd.DataFrame(rows)


def experiments_main(out_dir='outputs'):
    import os
    os.makedirs(out_dir, exist_ok=True)
    jobs = [('E1_illumination', e1_illumination), ('E2_scale', e2_scale),
            ('E3_ablation', e3_ablation), ('E4_noise', e4_noise)]
    all_df = []
    for name, fn in jobs:
        t = time.time()
        print(f'== {name} ==', flush=True)
        df = fn()
        df.to_csv(f'{out_dir}/{name}.csv', index=False)
        print(f'== {name}: {len(df)} runs in {time.time() - t:.0f}s ==', flush=True)
        all_df.append(df)
    pd.concat(all_df, ignore_index=True).to_csv(f'{out_dir}/all_runs.csv', index=False)
    print('done', flush=True)

# ==========================================================================
# analyse.py
# ==========================================================================

"""
analyse.py — aggregate the experiment CSVs and produce everything the dashboard
needs, as a single JSON payload.

Includes three measurements the dashboard renders live but that must be computed
here (they need the full image pipeline):

  * a fine solar-azimuth sweep of raw-intensity NCC vs phase-congruency NCC,
  * the MIM orientation-bin agreement matrix that demonstrates pi-periodicity,
  * base64 PNG strips of real renders, PC maps, tie points and checkerboards.
"""


import base64
import json
import os

import cv2
import numpy as np
import pandas as pd


VIEW = dict(rotation_deg=6.0, tx=6.0, ty=-4.0)


# --------------------------------------------------------------------------
# image helpers
# --------------------------------------------------------------------------

def _png_b64(arr: np.ndarray, cmap: str | None = None, size: int = 300) -> str:
    a = np.asarray(arr, np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    u8 = (a * 255).astype(np.uint8)
    u8 = cv2.resize(u8, (size, size), interpolation=cv2.INTER_AREA)
    if cmap == 'pc':
        # cold-blue -> warm-gold ramp, matching the dashboard palette
        lut = np.zeros((256, 3), np.uint8)
        for i in range(256):
            t = i / 255.0
            lut[i] = [int(30 + 200 * t ** 1.1),        # B
                      int(38 + 150 * t ** 1.3),        # G
                      int(46 + 190 * t ** 1.6)][::-1]  # R  (stored BGR)
        img = lut[u8]
    else:
        img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode('.png', img)
    return 'data:image/png;base64,' + base64.b64encode(buf).decode() if ok else ''


def _overlay_points(img: np.ndarray, pts: np.ndarray, size: int = 300,
                    colour=(90, 200, 240)) -> str:
    a = np.asarray(img, np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    u8 = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    sc = size / u8.shape[0]
    u8 = cv2.resize(u8, (size, size), interpolation=cv2.INTER_AREA)
    u8 = (u8 * 0.55).astype(np.uint8)
    for x, y in pts:
        cv2.circle(u8, (int(x * sc), int(y * sc)), 2, colour, -1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', u8)
    return 'data:image/png;base64,' + base64.b64encode(buf).decode() if ok else ''


# --------------------------------------------------------------------------
# live measurements
# --------------------------------------------------------------------------

def azimuth_sweep(step: int = 10, seed: int = 5) -> list[dict]:
    """Raw vs phase-congruency similarity as the sun rotates."""
    out = []
    for daz in range(0, 181, step):
        sp = auto_spec('OHRC', 'NAC', src_sun=(135.0, 35.0),
                       ref_sun=(135.0 + daz, 35.0),
                       view=ViewGeometry(), seed=seed)
        s, r, H, m = make_pair(sp)
        w = cv2.warpPerspective(s, H, (r.shape[1], r.shape[0]), flags=cv2.INTER_CUBIC)
        c = 40
        a, b = w[c:-c, c:-c], r[c:-c, c:-c]
        pa = phase_congruency(a)['M']
        pb = phase_congruency(b)['M']
        out.append(dict(
            d_az=daz,
            ncc_raw=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
            ncc_pc=float(np.corrcoef(pa.ravel(), pb.ravel())[0, 1]),
            grad_flip=float(gradient_flip_index(a, b)),
            src_png=_png_b64(a, size=220), ref_png=_png_b64(b, size=220),
            pc_src_png=_png_b64(pa, cmap='pc', size=220),
        ))
        print(f'  sweep daz={daz}', flush=True)
    return out


def mim_matrix(azimuths=(0, 30, 45, 60, 90, 120, 150, 180), seed: int = 5):
    """MIM agreement for every cyclic orientation-bin shift."""
    rows = []
    for daz in azimuths:
        sp = auto_spec('OHRC', 'NAC', src_sun=(135.0, 35.0),
                       ref_sun=(135.0 + daz, 35.0), view=ViewGeometry(), seed=seed)
        s, r, H, m = make_pair(sp)
        sw = cv2.resize(s, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)
        pa, pb = phase_congruency(sw), phase_congruency(r)
        ag = [float(((pa['MIM'] - k) % 6 == pb['MIM']).mean()) for k in range(6)]
        rows.append(dict(d_az=daz, agreement=ag,
                         best=int(np.argmax(ag)),
                         predicted=int(round((daz % 180) / 180.0 * 6)) % 6))
        print(f'  mim daz={daz}', flush=True)
    return rows


def showcase(seed: int = 5, daz: float = 120.0) -> dict:
    """One worked example, end to end, with imagery."""
    sp = auto_spec('OHRC', 'NAC', src_sun=(135.0, 35.0), ref_sun=(135.0 + daz, 22.0),
                   view=ViewGeometry(**VIEW), seed=seed)
    s, r, H, m = make_pair(sp)
    sw, Hpre = _prescale(s, 1.0 / m['scale_ratio'])
    H0 = coarse_align(sw, r)
    Hd, ps, pr, q = dense_register(sw, r, H0)
    e = true_geometric_error(Hd @ Hpre, H, s.shape)

    warped = warp_source(s, (Hd @ Hpre) / (Hd @ Hpre)[2, 2], r.shape)
    before = warp_source(s, np.eye(3), r.shape)

    # sparse comparison for the distribution panel
    o_sift = register(s, r, PipelineConfig(method='pcsift'),
                      scale_prior=m['scale_ratio'], H_true=H)

    ud = uniformity_report(pr, r.shape)
    return dict(
        d_az=daz, rmse_true=e['rmse_true'], max_true=e['max_true'],
        n_tiepoints=int(len(pr)),
        src_png=_png_b64(s), ref_png=_png_b64(r),
        pc_src_png=_png_b64(phase_congruency(sw)['M'], cmap='pc'),
        pc_ref_png=_png_b64(phase_congruency(r)['M'], cmap='pc'),
        warped_png=_png_b64(warped),
        cb_before_png=_png_b64(checkerboard(before, r)),
        cb_after_png=_png_b64(checkerboard(warped, r)),
        tiepoints_png=_overlay_points(r, pr),
        uniformity=ud,
        sparse_n=int(o_sift.get('n_final', 0) or 0),
        sparse_uniformity=o_sift.get('uniformity_after', {}),
        sparse_rmse=float(o_sift.get('rmse_true', float('nan'))),
    )


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def _sanitise(obj):
    """
    Replace NaN/Infinity with None throughout.

    json.dump emits bare NaN by default, which is NOT valid JSON. The browser's
    JSON.parse rejects it and throws before a single line of the dashboard runs,
    producing a blank page with no visible cause. Sanitising here (and passing
    allow_nan=False so the failure is loud rather than silent) is the fix.
    """
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _nanmed(series):
    a = series.to_numpy(float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else None


def _agg(df, group, value='rmse_true'):
    rows = []
    for keys, g in df.groupby(group):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ok = g['ok'].astype(bool).to_numpy()
        # Accuracy is quoted over CONVERGED runs only; mixing in the non-finite
        # errors of failed runs would silently change what the median means.
        v = g.loc[ok, value].to_numpy(float) if ok.any() else np.array([])
        finite = v[np.isfinite(v)]
        mean, lo, hi = bootstrap_ci(finite, np.median) if len(finite) >= 3 else (
            float(np.median(finite)) if len(finite) else float('nan'),
            float('nan'), float('nan'))
        rows.append(dict(zip(group if isinstance(group, list) else [group], keys)) | dict(
            n=int(len(g)), success_rate=float(ok.mean()),
            median=mean, ci_lo=lo, ci_hi=hi,
            n_ok=int(ok.sum()),
            median_n_final=_nanmed(g.loc[ok, 'n_final']),
            median_inlier_ratio=_nanmed(g.loc[ok, 'inlier_ratio']),
            median_time=_nanmed(g['t_total']),
        ))
    return rows


def analyse_main(out_dir='outputs'):
    payload = {}
    for name, group in [('E1_illumination', ['method', 'd_az']),
                        ('E2_scale', ['method', 'src_sensor', 'ref_sensor']),
                        ('E3_ablation', ['variant', 'd_az']),
                        ('E4_noise', ['method', 'snr_scale'])]:
        path = f'{out_dir}/{name}.csv'
        if not os.path.exists(path):
            print(f'  missing {path}', flush=True)
            continue
        df = pd.read_csv(path)
        payload[name] = _agg(df, group)
        payload[name + '_raw_n'] = int(len(df))
        print(f'  aggregated {name}: {len(df)} runs', flush=True)

        if name == 'E1_illumination':
            u = []
            for meth, g in df.groupby('method'):
                for col in ('unif_after_uniformity', 'unif_after_coverage',
                            'unif_after_clark_evans', 'unif_after_entropy'):
                    if col in g:
                        u.append(dict(method=meth, metric=col.replace('unif_after_', ''),
                                      value=float(np.nanmedian(g[col].to_numpy(float)))))
            payload['E1_uniformity'] = u

    print('azimuth sweep...', flush=True)
    payload['sweep'] = azimuth_sweep()
    print('mim matrix...', flush=True)
    payload['mim'] = mim_matrix()
    print('showcase...', flush=True)
    payload['showcase'] = showcase()

    with open(f'{out_dir}/dashboard_data.json', 'w') as f:
        json.dump(_sanitise(payload), f, allow_nan=False)
    size = os.path.getsize(f'{out_dir}/dashboard_data.json') / 1e6
    print(f'wrote dashboard_data.json ({size:.1f} MB)', flush=True)

# ==========================================================================
# stats.py
# ==========================================================================

"""
stats.py — the inferential layer.

Four analyses, each answering a question the raw tables cannot:

  1. FAILURE PREDICTION.  Given only quantities observable WITHOUT ground truth,
     can we tell that a converged registration is actually wrong? This is the
     operational question — on real PDS data there is no truth to check against.
     Measured on 159 converged runs of which 25 are silently wrong.

  2. DOSE-RESPONSE.  Fit success probability against solar azimuth difference
     per method and report the LD50 — the azimuth difference at which a method
     is a coin flip. This turns "SIFT stops working around 30-45°" into a number
     with a confidence interval.

  3. VARIANCE DECOMPOSITION.  Of the spread in registration error, how much is
     attributable to the method, to the illumination condition, and to which
     patch of terrain you happened to draw? If terrain dominates, comparing
     methods on a single scene is meaningless.

  4. PAIRED COMPARISON.  Methods are run on identical terrain-condition pairs,
     so differences should be tested paired, not as independent samples. A
     paired bootstrap respects that and gives an interval on the difference.

Small-sample discipline
-----------------------
25 positives is not many. A gradient-boosted model on 17 features will memorise
them, so every number here is cross-validated, and reported two ways: a random
stratified split (optimistic — related runs can straddle the fold boundary) and
a split grouped by terrain seed (honest — the model must generalise to unseen
ground). Where the two disagree, the grouped figure is the one to believe.
"""


import json
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

CSVS = ['E1_illumination', 'E2_scale', 'E3_ablation', 'E4_noise']

# Everything here is computable on real data with no reference truth.
OBSERVABLE = [
    'inlier_ratio', 'n_final', 'n_putative', 'n_inliers_raw', 'rmse_reproj',
    'resid_moran_x', 'resid_moran_y', 'resid_bias', 'resid_aniso',
    'unif_after_coverage', 'unif_after_entropy', 'unif_after_clark_evans',
    'unif_after_ripley_dev', 'unif_after_uniformity', 't_total',
    'rmse_ci_width', 'putative_to_final',
]

PRETTY = {
    'inlier_ratio': 'Inlier ratio',
    'n_final': 'Final tie points',
    'n_putative': 'Putative matches',
    'n_inliers_raw': 'Raw inliers',
    'rmse_reproj': 'Reprojection RMSE',
    'resid_moran_x': "Moran's I, x residuals",
    'resid_moran_y': "Moran's I, y residuals",
    'resid_bias': 'Residual mean bias',
    'resid_aniso': 'Residual anisotropy',
    'unif_after_coverage': 'Grid coverage',
    'unif_after_entropy': 'Cell entropy',
    'unif_after_clark_evans': 'Clark–Evans index',
    'unif_after_ripley_dev': 'Ripley K deviation',
    'unif_after_uniformity': 'Composite uniformity',
    't_total': 'Runtime',
    'rmse_ci_width': 'Bootstrap CI width',
    'putative_to_final': 'Survival fraction',
}


def load(out_dir='outputs') -> pd.DataFrame:
    frames = []
    for name in CSVS:
        try:
            frames.append(pd.read_csv(f'{out_dir}/{name}.csv'))
        except FileNotFoundError:
            continue
    df = pd.concat(frames, ignore_index=True)
    df['rmse_ci_width'] = df['rmse_ci_hi'] - df['rmse_ci_lo']
    df['putative_to_final'] = df['n_final'] / df['n_putative'].replace(0, np.nan)
    return df


# --------------------------------------------------------------------------
# 1. failure prediction
# --------------------------------------------------------------------------

def failure_dataset(df: pd.DataFrame, threshold: float = 1.0):
    """Converged runs only — the population where a silent failure is possible."""
    d = df[df['ok'].astype(bool) & df['rmse_true'].notna()].copy()
    y = (d['rmse_true'] > threshold).astype(int).to_numpy()
    X = d[OBSERVABLE].to_numpy(float)
    groups = d['seed'].fillna(-1).to_numpy()
    return X, y, groups, d


def _cv_scores(X, y, groups, model_fn, mode='stratified', n_splits=5, seed=0):
    """Out-of-fold probabilities, so every score is on data the model never saw."""
    oof = np.full(len(y), np.nan)
    if mode == 'grouped':
        uniq = np.unique(groups)
        n_splits = int(min(n_splits, len(uniq)))
        if n_splits < 2:
            return oof
        splitter = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed).split(X, y)
    for tr, te in splitter:
        if len(np.unique(y[tr])) < 2:
            continue
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def _auc_ci(y, p, n_boot=2000, seed=0):
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return None, None, None
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, p)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        bs.append(roc_auc_score(y[i], p[i]))
    return float(base), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _logreg():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=0.4, max_iter=3000,
                                            class_weight='balanced'))


def _gbm():
    return make_pipeline(SimpleImputer(strategy='median'),
                         GradientBoostingClassifier(n_estimators=120, max_depth=2,
                                                    learning_rate=0.06,
                                                    subsample=0.85, random_state=0))


def failure_analysis(df: pd.DataFrame, threshold: float = 1.0) -> dict:
    X, y, groups, d = failure_dataset(df, threshold)
    out = dict(n=int(len(y)), n_fail=int(y.sum()),
               base_rate=float(y.mean()), threshold=threshold)

    # A confound to rule out before believing any of this: silent failures are
    # concentrated in particular configurations (no-scale-prior runs, the sparse
    # matchers), and count features like "number of putative matches" are strong
    # signatures of WHICH method ran. A model could score well by recognising
    # the configuration rather than by detecting error. Splitting on
    # configuration forces it to generalise to setups it has never seen.
    config = (d['method'].astype(str) + '|' + d['variant'].fillna('-').astype(str) +
              '|' + d['src_sensor'].fillna('-').astype(str) +
              d['ref_sensor'].fillna('-').astype(str)).to_numpy().astype(str)

    models = {'logistic': _logreg, 'gbm': _gbm}
    out['models'] = {}
    best = None
    for name, fn in models.items():
        entry = {}
        for mode in ('stratified', 'grouped', 'config'):
            g = config if mode == 'config' else groups
            p = _cv_scores(X, y, g, fn, mode='grouped' if mode != 'stratified' else 'stratified')
            a, lo, hi = _auc_ci(y, p)
            entry[mode] = dict(auc=a, lo=lo, hi=hi)
            # Operating curves are reported from the CONFIG split — the only one
            # that forces generalisation to unseen setups. Measured: the gradient
            # boosting model scores 0.990 on a random split and 0.467 (chance) on
            # this one, i.e. it was recognising configurations, not detecting
            # error. The regularised linear model holds at 0.890, so that is the
            # model that gets deployed.
            if mode == 'config' and a is not None:
                if best is None or a > best[1]:
                    best = (name, a, p)
        out['models'][name] = entry

    # single-feature AUCs: which diagnostics carry signal on their own?
    singles = []
    for j, f in enumerate(OBSERVABLE):
        v = X[:, j]
        ok = np.isfinite(v)
        if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
            continue
        a = roc_auc_score(y[ok], v[ok])
        singles.append(dict(feature=f, label=PRETTY.get(f, f),
                            auc=float(max(a, 1 - a)),
                            direction='higher→failure' if a > 0.5 else 'lower→failure',
                            rho=float(sps.spearmanr(v[ok], y[ok]).statistic)))
    singles.sort(key=lambda r: -r['auc'])
    out['single_feature'] = singles

    # Hardest test available: restrict to the dense pipeline alone, so method
    # identity carries no information and the model must judge runs of the SAME
    # configuration against each other.
    dm = (d['method'] == 'dense').to_numpy()
    if dm.sum() > 30 and 0 < y[dm].sum() < dm.sum():
        Xd, yd, gd = X[dm], y[dm], groups[dm]
        pd_ = _cv_scores(Xd, yd, gd, _logreg, mode='grouped')
        a, lo, hi = _auc_ci(yd, pd_)
        out['within_dense'] = dict(n=int(dm.sum()), n_fail=int(yd.sum()),
                                   auc=a, lo=lo, hi=hi)
        cfg_d = config[dm]
        pc_ = _cv_scores(Xd, yd, cfg_d, _logreg, mode='grouped')
        a2, lo2, hi2 = _auc_ci(yd, pc_)
        out['within_dense']['auc_config'] = a2
        out['within_dense']['lo_config'] = lo2
        out['within_dense']['hi_config'] = hi2
    else:
        out['within_dense'] = None

    # operating curve of the best cross-validated model
    if best is not None:
        name, auc, p = best
        ok = np.isfinite(p)
        yv, pv = y[ok], p[ok]
        fpr, tpr, thr = roc_curve(yv, pv)
        step = max(1, len(fpr) // 60)
        out['roc'] = dict(model=name,
                          fpr=[float(v) for v in fpr[::step]],
                          tpr=[float(v) for v in tpr[::step]])
        # full sweep so the dashboard can move the threshold live
        grid = np.linspace(0.02, 0.98, 49)
        sweep = []
        for t in grid:
            pred = (pv >= t).astype(int)
            tp = int(((pred == 1) & (yv == 1)).sum())
            fp = int(((pred == 1) & (yv == 0)).sum())
            fn = int(((pred == 0) & (yv == 1)).sum())
            tn = int(((pred == 0) & (yv == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else None
            rec = tp / (tp + fn) if tp + fn else None
            f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else 0.0
            sweep.append(dict(t=float(t), tp=tp, fp=fp, fn=fn, tn=tn,
                              precision=prec, recall=rec, f1=float(f1)))
        out['sweep'] = sweep
        out['scores'] = [dict(p=float(a), y=int(b)) for a, b in zip(pv, yv)]

        # calibration in quantile bins
        try:
            qs = np.quantile(pv, np.linspace(0, 1, 6))
            qs = np.unique(qs)
            cal = []
            for a, b in zip(qs[:-1], qs[1:]):
                m = (pv >= a) & (pv <= b)
                if m.sum() >= 5:
                    cal.append(dict(pred=float(pv[m].mean()),
                                    obs=float(yv[m].mean()), n=int(m.sum())))
            out['calibration'] = cal
        except Exception:
            out['calibration'] = []
    return out


# --------------------------------------------------------------------------
# 2. dose-response
# --------------------------------------------------------------------------

def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(k * (x - x0)))


def dose_response(df: pd.DataFrame, n_boot: int = 600) -> list:
    """Success probability vs solar azimuth difference, per method, with LD50."""
    d = df[(df['experiment'] == 'E1_illumination') & df['d_az'].notna()]
    out = []
    for method, g in d.groupby('method'):
        x = g['d_az'].to_numpy(float)
        y = g['ok'].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            out.append(dict(method=method, ld50=None, lo=None, hi=None,
                            points=_binned(x, y), always=bool(y.all())))
            continue
        try:
            popt, _ = optimize.curve_fit(_logistic, x, y, p0=[60.0, 0.08],
                                         maxfev=20000)
        except Exception:
            popt = [np.nan, np.nan]
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(x), len(x))
            if len(np.unique(y[i])) < 2:
                continue
            try:
                p, _ = optimize.curve_fit(_logistic, x[i], y[i], p0=popt if
                                          np.isfinite(popt[0]) else [60.0, 0.08],
                                          maxfev=8000)
                if 0 <= p[0] <= 360:
                    bs.append(p[0])
            except Exception:
                continue
        curve = [dict(x=float(v), y=float(_logistic(v, *popt)))
                 for v in np.linspace(0, 180, 61)] if np.isfinite(popt[0]) else []
        out.append(dict(
            method=method,
            ld50=float(popt[0]) if np.isfinite(popt[0]) else None,
            lo=float(np.percentile(bs, 2.5)) if len(bs) > 30 else None,
            hi=float(np.percentile(bs, 97.5)) if len(bs) > 30 else None,
            curve=curve, points=_binned(x, y), always=bool(y.all())))
    return out


def _binned(x, y):
    out = []
    for v in np.unique(x):
        m = x == v
        k, n = int(y[m].sum()), int(m.sum())
        # Wilson interval — correct for proportions near 0 and 1, unlike normal
        lo, hi = _wilson(k, n)
        out.append(dict(x=float(v), p=k / n, n=n, lo=lo, hi=hi))
    return out


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (float(max(0, ctr - half)), float(min(1, ctr + half)))


# --------------------------------------------------------------------------
# 3. variance decomposition
# --------------------------------------------------------------------------

def variance_decomposition(df: pd.DataFrame) -> dict:
    """
    How much of the spread in log error is method, condition, terrain, residual?

    Errors span orders of magnitude, so the decomposition runs on log10(error);
    on the raw scale a single 400 px outlier would swamp every other term.
    """
    d = df[(df['experiment'] == 'E1_illumination') & df['ok'].astype(bool) &
           df['rmse_true'].notna()].copy()
    d['le'] = np.log10(d['rmse_true'].clip(lower=1e-3))
    grand = d['le'].mean()
    total = ((d['le'] - grand) ** 2).sum()
    if total <= 0:
        return dict(components=[], n=int(len(d)))

    comps = []
    for name, col in [('Method', 'method'), ('Illumination Δaz', 'd_az'),
                      ('Terrain draw', 'seed')]:
        ss = sum(len(g) * (g['le'].mean() - grand) ** 2 for _, g in d.groupby(col))
        comps.append(dict(name=name, share=float(ss / total)))
    explained = sum(c['share'] for c in comps)
    comps.append(dict(name='Unexplained', share=float(max(0.0, 1 - explained))))
    return dict(components=comps, n=int(len(d)),
                sd_log10=float(d['le'].std()))


# --------------------------------------------------------------------------
# 4. paired comparison
# --------------------------------------------------------------------------

def paired_comparison(df: pd.DataFrame, n_boot: int = 4000) -> list:
    """
    Dense against each baseline on identical (terrain, illumination) cells.

    Unpaired tests would attribute to the method a difference that is really the
    luck of the terrain draw, which the variance decomposition shows is large.
    """
    d = df[df['experiment'] == 'E1_illumination'].copy()
    piv = d.pivot_table(index=['seed', 'd_az'], columns='method',
                        values='ok', aggfunc='max')
    out = []
    if 'dense' not in piv.columns:
        return out
    rng = np.random.default_rng(0)
    for m in piv.columns:
        if m == 'dense':
            continue
        sub = piv[['dense', m]].dropna()
        if len(sub) < 5:
            continue
        a = sub['dense'].to_numpy(float)
        b = sub[m].to_numpy(float)
        diff = a - b
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean()
              for _ in range(n_boot)]
        # McNemar on discordant cells
        n01 = int(((a == 0) & (b == 1)).sum())
        n10 = int(((a == 1) & (b == 0)).sum())
        pval = float(sps.binomtest(n10, n10 + n01, 0.5).pvalue) if (n10 + n01) else 1.0
        out.append(dict(method=m, n_pairs=int(len(sub)),
                        delta=float(diff.mean()),
                        lo=float(np.percentile(bs, 2.5)),
                        hi=float(np.percentile(bs, 97.5)),
                        wins=n10, losses=n01, p=pval))
    out.sort(key=lambda r: -r['delta'])
    return out


# --------------------------------------------------------------------------

def run_explorer_rows(df: pd.DataFrame, max_rows: int = 900) -> list:
    d = df[df['ok'].astype(bool) & df['rmse_true'].notna()].copy()
    cols = OBSERVABLE + ['rmse_true', 'method', 'd_az', 'experiment']
    d = d[[c for c in cols if c in d.columns]]
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=0)
    recs = []
    for _, r in d.iterrows():
        rec = {}
        for c in d.columns:
            v = r[c]
            if isinstance(v, (int, float, np.floating, np.integer)):
                v = float(v)
                rec[c] = None if not np.isfinite(v) else v
            else:
                rec[c] = v
        recs.append(rec)
    return recs


def stats_main(out_dir='outputs'):
    df = load(out_dir)
    payload = dict(
        failure=failure_analysis(df),
        dose=dose_response(df),
        variance=variance_decomposition(df),
        paired=paired_comparison(df),
        runs=run_explorer_rows(df),
        features=[dict(key=k, label=PRETTY[k]) for k in OBSERVABLE],
        n_total=int(len(df)),
    )
    with open(f'{out_dir}/stats.json', 'w') as f:
        json.dump(payload, f, allow_nan=False)
    return payload

# ==========================================================================
# build_dashboard.py
# ==========================================================================

"""
build_dashboard.py — inject the measurement payload into the HTML template.

Validates before writing. The check matters: json.dump emits bare NaN by
default, which is not valid JSON, and JSON.parse rejects it before a single
line of the page runs — producing a blank page with no visible cause. Note the
regex word boundaries: a naive `'NaN' not in data` check gives false positives,
because "NaN" occurs inside base64 image data.
"""


import json
import os
import re
import sys

BARE = re.compile(r'(?<![\"\w])(NaN|Infinity|-Infinity)(?![\"\w])')


def build(template='outputs/dashboard_template.html',
          data='outputs/dashboard_data.json',
          out='outputs/lunar_registration_dashboard.html') -> str:
    tpl = open(template, encoding='utf-8').read()
    payload = open(data, encoding='utf-8').read()

    if '__DATA__' not in tpl:
        raise SystemExit(f'{template}: __DATA__ placeholder missing')
    bad = BARE.findall(payload)
    if bad:
        raise SystemExit(f'{data}: {len(bad)} invalid JSON token(s), e.g. {bad[:3]}. '
                         'Re-run `python -m lunareg.analyse`.')
    json.loads(payload)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(tpl.replace('__DATA__', payload))
    print(f'{out}  {os.path.getsize(out) / 1e6:.2f} MB')
    return out

# ==========================================================================
# cli.py
# ==========================================================================

"""
cli.py — command-line entry point (`lunareg ...`).

Subcommands
-----------
register     Register one real (or PNG/TIFF) source image onto a reference
             image and write the product deliverable: warped image,
             checkerboard QC mosaic, and a product.json with the homography,
             every match point, and the evaluation metrics.
experiments  Re-run the E1-E4 synthetic validation study (lunareg.experiments).
analyse      Aggregate experiment CSVs into the dashboard's JSON payload.
dashboard    Inject that payload into the HTML dashboard template.
"""


import argparse
import json
import os
import sys

import cv2
import numpy as np


def _load_any(path: str, gsd_override: float | None):
    """Load a real product via lunareg.io, or a plain image if that fails."""
    try:
        li = load_image(path)
        if gsd_override is not None:
            li.meta.gsd_m = gsd_override
        return li.data, li.meta
    except Exception as e:  # noqa: BLE001 - fall back to a plain image read
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f'{path}: not a recognised mission product and not '
                             f'readable as a plain image ({e})') from e
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        meta = LunarImageMeta(path=path, format='raster', lines=arr.shape[0],
                                  samples=arr.shape[1], gsd_m=gsd_override,
                                  notes=[f'loaded as a plain image ({e})'])
        return arr.astype(np.float32), meta


def cmd_register(args):

    src, src_meta = _load_any(args.src, args.src_gsd)
    ref, ref_meta = _load_any(args.ref, args.ref_gsd)
    for meta in (src_meta, ref_meta):
        for note in meta.notes:
            print(f'[warn] {meta.path}: {note}', file=sys.stderr)

    prior = scale_prior(src_meta, ref_meta) if not args.no_scale_prior else None
    d_az = sun_azimuth_delta(src_meta, ref_meta)
    if prior is None and not args.no_scale_prior:
        print('[warn] no scale prior available from metadata; matcher will '
             'solve for scale (pass --src-gsd/--ref-gsd to supply one)',
             file=sys.stderr)

    cfg = PipelineConfig(method=args.method, use_scale_prior=not args.no_scale_prior,
                         n_features=args.n_features, target_points=args.target_points,
                         d_azimuth_prior=d_az)
    out = register(src, ref, cfg, scale_prior=prior, verbose=args.verbose)

    os.makedirs(args.out_dir, exist_ok=True)
    if not out.get('ok'):
        print(f"registration FAILED: {out.get('fail', 'unknown')}", file=sys.stderr)
        with open(os.path.join(args.out_dir, 'product.json'), 'w') as f:
            json.dump({k: v for k, v in out.items() if k != 'H'}, f, indent=2)
        return 1

    H = out['H']
    warped = warp_source(src, H, ref.shape)
    cb = checkerboard(warped, ref)

    def _save(name, img):
        lo, hi = np.percentile(img, [0.5, 99.5])
        u8 = np.clip((img - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out_dir, name), u8)

    _save('warped_source.png', warped)
    _save('checkerboard.png', cb)

    product = {
        'src_path': args.src, 'ref_path': args.ref, 'method': out['method'],
        'homography_src_to_ref': H.tolist(),
        'model_used': out.get('model_used'),
        'metrics': {
            'rmse_reproj_px': out.get('rmse_reproj'),
            'rmse_ci_95': list(out.get('rmse_ci', (None, None))),
            'inlier_ratio': out.get('inlier_ratio'),
            'n_putative_matches': out.get('n_putative'),
            'n_inlier_matches': out.get('n_final'),
            'uniformity': out.get('uniformity_after'),
        },
        'timing_s': out.get('t_total'),
    }
    with open(os.path.join(args.out_dir, 'product.json'), 'w') as f:
        json.dump(product, f, indent=2)

    print(f"OK  method={out['method']}  model={out.get('model_used')}  "
         f"inliers={out.get('n_final')}/{out.get('n_putative')}  "
         f"rmse={out.get('rmse_reproj'):.3f}px  -> {args.out_dir}/")
    return 0


def cmd_experiments(args):
    experiments_main(args.out_dir)
    return 0


def cmd_analyse(args):
    analyse_main(args.out_dir)
    return 0


def cmd_dashboard(args):
    build(
        template=os.path.join(args.out_dir, 'dashboard_template.html'),
        data=os.path.join(args.out_dir, 'dashboard_data.json'),
        out=os.path.join(args.out_dir, 'lunar_registration_dashboard.html'))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='lunareg', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    r = sub.add_parser('register', help='register a source image onto a reference image')
    r.add_argument('--src', required=True, help='source (moving) image: .img/.xml/.lbl/.tif/.png')
    r.add_argument('--ref', required=True, help='reference (fixed) image')
    r.add_argument('--src-gsd', type=float, default=None, help='override source GSD, metres/px')
    r.add_argument('--ref-gsd', type=float, default=None, help='override reference GSD, metres/px')
    r.add_argument('--method', default='hybrid',
                  choices=['rift', 'sift', 'pcsift', 'akaze', 'orb', 'dense', 'hybrid'])
    r.add_argument('--no-scale-prior', action='store_true',
                  help='ignore metadata GSDs and let the matcher solve for scale')
    r.add_argument('--n-features', type=int, default=2500)
    r.add_argument('--target-points', type=int, default=400)
    r.add_argument('--out-dir', default='outputs/register')
    r.add_argument('--verbose', action='store_true')
    r.set_defaults(func=cmd_register)

    e = sub.add_parser('experiments', help='run the synthetic validation study (E1-E4)')
    e.add_argument('--out-dir', default='outputs')
    e.set_defaults(func=cmd_experiments)

    a = sub.add_parser('analyse', help='aggregate experiment CSVs into dashboard_data.json')
    a.add_argument('--out-dir', default='outputs')
    a.set_defaults(func=cmd_analyse)

    d = sub.add_parser('dashboard', help='build the HTML dashboard from dashboard_data.json')
    d.add_argument('--out-dir', default='outputs')
    d.set_defaults(func=cmd_dashboard)

    return p


def cli_main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0

if __name__ == '__main__':
    raise SystemExit(cli_main())
