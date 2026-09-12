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

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .matching import _upsampled_dft_shift, estimate_model, estimate_progressive
from .photometric import phase_congruency


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
