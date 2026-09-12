"""
analyse.py — aggregate the experiment CSVs and produce everything the dashboard
needs, as a single JSON payload.

Includes three measurements the dashboard renders live but that must be computed
here (they need the full image pipeline):

  * a fine solar-azimuth sweep of raw-intensity NCC vs phase-congruency NCC,
  * the MIM orientation-bin agreement matrix that demonstrates pi-periodicity,
  * base64 PNG strips of real renders, PC maps, tie points and checkerboards.
"""

from __future__ import annotations

import base64
import io
import json
import os

import cv2
import numpy as np
import pandas as pd

from .dense import coarse_align, dense_register, grid_tiepoints
from .metrics import apply_H, bootstrap_ci, true_geometric_error
from .photometric import gradient_flip_index, phase_congruency
from .pipeline import PipelineConfig, _prescale, checkerboard, register, warp_source
from .synth import ViewGeometry, auto_spec, make_pair
from . import distribution as dist

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

    ud = dist.uniformity_report(pr, r.shape)
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


def main(out_dir='outputs'):
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


if __name__ == '__main__':
    main()
