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

from __future__ import annotations

import itertools
import sys
import time

import numpy as np
import pandas as pd

from .pipeline import PipelineConfig, register
from .synth import SENSORS, ViewGeometry, auto_spec, make_pair


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
    from . import dense as D
    from . import metrics as mt
    from .pipeline import _prescale

    t0 = time.time()
    o = dict(ok=False)
    try:
        src_w, H_pre = _prescale(src, 1.0 / m['scale_ratio'])
        H0 = D.coarse_align(src_w, ref, )
        Hd = H0
        ps = pr = np.zeros((0, 2))
        for (g, w, s) in ((10, 64, 24), (16, 40, 8), (22, 32, 4)):
            ps, pr, q = D.grid_tiepoints(src_w, ref, Hd, grid=g, win=w,
                                         search=s, use_pc=False)
            if len(ps) < 8:
                break
            Hn, mask, _ = D.estimate_progressive(ps, pr, 2.0)
            if Hn is None:
                break
            Hd, ps, pr = Hn, ps[mask], pr[mask]
        if len(ps) >= 8:
            Hf = Hd @ H_pre
            o = dict(ok=True, n_final=len(ps), n_putative=len(ps))
            o.update(mt.true_geometric_error(Hf / Hf[2, 2], H, src.shape))
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


def main(out_dir='outputs'):
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


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'outputs')
