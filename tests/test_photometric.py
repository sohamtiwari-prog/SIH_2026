import numpy as np

from lunareg import features as feat
from lunareg import photometric as ph
from lunareg import synth


def test_phase_congruency_output_shapes_and_ranges(rng):
    img = rng.normal(0, 1, (128, 128))
    pc = ph.phase_congruency(img, nscale=3, norient=4)
    for key in ('M', 'm'):
        assert pc[key].shape == img.shape
        assert pc[key].min() >= 0.0 and pc[key].max() <= 1.0 + 1e-6
    assert pc['MIM'].shape == img.shape
    assert pc['MIM'].min() >= 0 and pc['MIM'].max() < pc['norient']
    assert pc['norient'] == 4


def test_sun_bin_shift_periodicity_and_symmetry():
    norient = 6
    assert feat.sun_bin_shift(0.0, norient) == 0
    # a 180 degree azimuth change must map back to zero shift (pi-periodicity)
    assert feat.sun_bin_shift(180.0, norient) == 0
    assert feat.sun_bin_shift(90.0, norient) == feat.sun_bin_shift(-90.0, norient)


def test_gradient_flip_index_near_one_for_identical_images(rng):
    img = rng.normal(0, 1, (100, 100))
    from scipy import ndimage
    img = ndimage.gaussian_filter(img, 2.0)
    assert ph.gradient_flip_index(img, img) > 0.99


def test_phase_congruency_more_illumination_stable_than_raw_intensity():
    """The core claim of photometric.py: PC correlates across a sun flip far
    better than raw intensity does. Regression-tests that gap on a real
    physically-rendered pair rather than asserting it in the abstract."""
    tc = synth.TerrainConfig(n=192, n_craters=40, seed=3)
    dem, alb, _ = synth.make_terrain(tc)
    a = synth.render(dem, alb, tc.gsd_m, sun_az_deg=120.0, sun_el_deg=35.0)
    b = synth.render(dem, alb, tc.gsd_m, sun_az_deg=300.0, sun_el_deg=35.0)  # 180 deg flip

    ncc_raw = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    pa, pb = ph.phase_congruency(a)['M'], ph.phase_congruency(b)['M']
    ncc_pc = float(np.corrcoef(pa.ravel(), pb.ravel())[0, 1])

    assert ncc_pc > ncc_raw
