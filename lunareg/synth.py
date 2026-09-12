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

from __future__ import annotations

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
