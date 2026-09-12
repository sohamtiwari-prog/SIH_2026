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

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np


def _load_any(path: str, gsd_override: float | None):
    """Load a real product via lunareg.io, or a plain image if that fails."""
    from . import io as lio
    try:
        li = lio.load_image(path)
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
        meta = lio.LunarImageMeta(path=path, format='raster', lines=arr.shape[0],
                                  samples=arr.shape[1], gsd_m=gsd_override,
                                  notes=[f'loaded as a plain image ({e})'])
        return arr.astype(np.float32), meta


def cmd_register(args):
    from . import io as lio
    from .pipeline import PipelineConfig, register, warp_source, checkerboard

    src, src_meta = _load_any(args.src, args.src_gsd)
    ref, ref_meta = _load_any(args.ref, args.ref_gsd)
    for meta in (src_meta, ref_meta):
        for note in meta.notes:
            print(f'[warn] {meta.path}: {note}', file=sys.stderr)

    prior = lio.scale_prior(src_meta, ref_meta) if not args.no_scale_prior else None
    d_az = lio.sun_azimuth_delta(src_meta, ref_meta)
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
    from . import experiments
    experiments.main(args.out_dir)
    return 0


def cmd_analyse(args):
    from . import analyse
    analyse.main(args.out_dir)
    return 0


def cmd_dashboard(args):
    from . import build_dashboard
    build_dashboard.build(
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


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
