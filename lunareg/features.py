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

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .photometric import phase_congruency


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
