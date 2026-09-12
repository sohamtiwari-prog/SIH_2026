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

from __future__ import annotations

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
