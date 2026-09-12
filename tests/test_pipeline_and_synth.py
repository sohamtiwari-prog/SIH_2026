import numpy as np
import pytest

from lunareg.synth import auto_spec, make_pair, ViewGeometry
from lunareg.pipeline import PipelineConfig, register, warp_source, checkerboard
from lunareg.metrics import apply_H


@pytest.fixture(scope='module')
def synthetic_pair():
    """One synthetic OHRC->NAC pair: moderate sun-angle and viewpoint change,
    with the exact ground-truth homography synth.py derives analytically."""
    sp = auto_spec('OHRC', 'NAC', src_sun=(135.0, 35.0), ref_sun=(225.0, 25.0),
                   view=ViewGeometry(rotation_deg=6.0, tx=6.0, ty=-4.0),
                   coarse_px=160, max_px=256, seed=7)
    return make_pair(sp)


def test_make_pair_H_true_maps_centre_consistently(synthetic_pair):
    src, ref, H_true, meta = synthetic_pair
    assert src.dtype == np.float32 and ref.dtype == np.float32
    # the transform must be a proper (non-degenerate) homography
    assert abs(np.linalg.det(H_true)) > 1e-9
    centre = np.array([[src.shape[1] / 2, src.shape[0] / 2]])
    mapped = apply_H(H_true, centre)[0]
    assert 0 <= mapped[0] <= ref.shape[1]
    assert 0 <= mapped[1] <= ref.shape[0]


def test_register_dense_recovers_ground_truth_within_subpixel(synthetic_pair):
    src, ref, H_true, meta = synthetic_pair
    cfg = PipelineConfig(method='dense')
    out = register(src, ref, cfg, scale_prior=meta['scale_ratio'], H_true=H_true)
    assert out['ok'], out.get('fail')
    assert out['rmse_true'] < 1.0  # sub-pixel accuracy requirement from the PS
    assert out['n_final'] >= 8


def test_register_reports_failure_reason_on_degenerate_input():
    blank_src = np.zeros((64, 64), np.float32)
    blank_ref = np.zeros((64, 64), np.float32)
    out = register(blank_src, blank_ref, PipelineConfig(method='dense'))
    assert not out['ok']
    assert 'fail' in out


def test_warp_source_and_checkerboard_shapes(synthetic_pair):
    src, ref, H_true, meta = synthetic_pair
    warped = warp_source(src, H_true, ref.shape)
    assert warped.shape == ref.shape
    cb = checkerboard(warped, ref)
    assert cb.shape == ref.shape
