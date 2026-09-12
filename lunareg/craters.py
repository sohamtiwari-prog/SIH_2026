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

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .photometric import phase_congruency


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
