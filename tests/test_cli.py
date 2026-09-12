import json
import os

import cv2
import numpy as np

from lunareg.cli import main
from lunareg.synth import auto_spec, make_pair, ViewGeometry


def _save_u8(path, img):
    lo, hi = np.percentile(img, [0.5, 99.5])
    u8 = np.clip((img - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, u8)


def test_register_command_end_to_end(tmp_path):
    sp = auto_spec('OHRC', 'NAC', src_sun=(135.0, 35.0), ref_sun=(225.0, 25.0),
                   view=ViewGeometry(rotation_deg=6.0, tx=6.0, ty=-4.0),
                   coarse_px=160, max_px=256, seed=11)
    src, ref, H_true, meta = make_pair(sp)
    src_path, ref_path = str(tmp_path / 'src.png'), str(tmp_path / 'ref.png')
    _save_u8(src_path, src)
    _save_u8(ref_path, ref)
    out_dir = str(tmp_path / 'out')

    rc = main(['register', '--src', src_path, '--ref', ref_path,
              '--method', 'dense', '--no-scale-prior', '--out-dir', out_dir])

    assert rc == 0
    assert os.path.exists(os.path.join(out_dir, 'product.json'))
    assert os.path.exists(os.path.join(out_dir, 'warped_source.png'))
    assert os.path.exists(os.path.join(out_dir, 'checkerboard.png'))

    with open(os.path.join(out_dir, 'product.json')) as f:
        product = json.load(f)
    assert len(product['homography_src_to_ref']) == 3
    assert product['metrics']['rmse_reproj_px'] is not None
    assert product['metrics']['n_inlier_matches'] > 0


def test_register_command_reports_failure_without_crashing(tmp_path):
    blank = np.zeros((64, 64), np.uint8)
    src_path, ref_path = str(tmp_path / 'blank_src.png'), str(tmp_path / 'blank_ref.png')
    cv2.imwrite(src_path, blank)
    cv2.imwrite(ref_path, blank)
    out_dir = str(tmp_path / 'out')

    rc = main(['register', '--src', src_path, '--ref', ref_path,
              '--method', 'dense', '--out-dir', out_dir])

    assert rc == 1
    with open(os.path.join(out_dir, 'product.json')) as f:
        product = json.load(f)
    assert product['ok'] is False
