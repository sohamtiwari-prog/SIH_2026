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

from __future__ import annotations

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
    from .photometric import phase_congruency

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
