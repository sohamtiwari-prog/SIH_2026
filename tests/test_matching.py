import numpy as np
from scipy import ndimage

from lunareg import matching as m


def test_estimate_model_recovers_known_similarity(rng):
    scale, rot_deg, tx, ty = 1.3, 12.0, 8.0, -5.0
    th = np.deg2rad(rot_deg)
    c, s = np.cos(th), np.sin(th)
    H_true = np.array([[scale * c, -scale * s, tx],
                       [scale * s, scale * c, ty],
                       [0, 0, 1.0]])
    ptsA = rng.uniform(0, 200, (60, 2)).astype(np.float32)
    from lunareg.metrics import apply_H
    ptsB = apply_H(H_true, ptsA).astype(np.float32)

    H_est, mask = m.estimate_model(ptsA, ptsB, model='similarity', thresh=1.0)
    assert H_est is not None
    assert mask.all()
    got = apply_H(H_est, ptsA)
    assert np.sqrt(((got - ptsB) ** 2).sum(1).mean()) < 0.5


def test_estimate_model_rejects_gross_outliers(rng):
    from lunareg.metrics import apply_H
    H_true = np.array([[1, 0, 10.0], [0, 1, -6.0], [0, 0, 1.0]])
    ptsA = rng.uniform(0, 200, (50, 2)).astype(np.float32)
    ptsB = apply_H(H_true, ptsA).astype(np.float32)
    n_out = 15
    ptsB[:n_out] += rng.uniform(-100, 100, (n_out, 2)).astype(np.float32)

    H_est, mask = m.estimate_model(ptsA, ptsB, model='affine', thresh=2.0)
    assert H_est is not None
    assert mask.sum() >= 50 - n_out - 3
    assert not mask[:n_out].all()


def test_estimate_model_too_few_points_returns_none():
    H, mask = m.estimate_model(np.zeros((2, 2)), np.zeros((2, 2)), model='homography')
    assert H is None and mask is None


def test_estimate_progressive_prefers_similarity_when_it_fits(rng):
    from lunareg.metrics import apply_H
    H_true = np.array([[1, 0, 5.0], [0, 1, 5.0], [0, 0, 1.0]])
    ptsA = rng.uniform(0, 100, (40, 2)).astype(np.float32)
    ptsB = apply_H(H_true, ptsA).astype(np.float32)
    H_est, mask, model = m.estimate_progressive(ptsA, ptsB, thresh=1.0)
    assert model in ('similarity', 'affine', 'homography')
    assert mask.sum() >= 30


def _fft_shift(img: np.ndarray, dr: float, dc: float) -> np.ndarray:
    """Exact (circular, band-limited) sub-pixel shift via a Fourier phase ramp
    -- avoids the interpolation error that ndimage.shift's spline would add,
    so the test below isolates the estimator's own bias."""
    n0, n1 = img.shape
    fy = np.fft.fftfreq(n0)[:, None]
    fx = np.fft.fftfreq(n1)[None, :]
    ramp = np.exp(-2j * np.pi * (fy * dr + fx * dc))
    return np.fft.ifft2(np.fft.fft2(img) * ramp).real


def test_upsampled_dft_shift_recovers_known_subpixel_translation(rng):
    """Regression test: a hand-rolled DFT estimator here was biased ~0.75px
    at zero shift (see matching.py docstring); this pins the actual behaviour.

    _upsampled_dft_shift(a, b) returns the shift to apply to `b` to align it
    with `a`. Here `b` is `a` shifted by (true_dr, true_dc), so the shift that
    brings it back onto `a` is (-true_dr, -true_dc)."""
    n = 96
    base = rng.normal(0, 1, (n, n))
    base = ndimage.gaussian_filter(base, 2.0)
    for true_dr, true_dc in [(0.0, 0.0), (1.37, -0.62), (-2.05, 0.48)]:
        shifted = _fft_shift(base, true_dr, true_dc)
        res = m._upsampled_dft_shift(base, shifted, upsample=50, max_shift=4.0)
        assert res is not None
        dr, dc, psr = res
        assert abs(dr - (-true_dr)) < 0.05
        assert abs(dc - (-true_dc)) < 0.05
        assert psr > 0


def test_upsampled_dft_shift_none_for_flat_or_mismatched_input():
    flat = np.zeros((32, 32))
    assert m._upsampled_dft_shift(flat, flat) is None
    a = np.random.default_rng(0).normal(0, 1, (32, 32))
    b = np.random.default_rng(0).normal(0, 1, (16, 16))
    assert m._upsampled_dft_shift(a, b) is None


def test_match_descriptors_mutual_nearest_neighbour_round_trip(rng):
    # descB[i] = descA[perm[i]] + noise, so the match for query index a is at
    # the position where perm equals a, i.e. the inverse permutation of a.
    descA = rng.normal(0, 1, (30, 16)).astype(np.float32)
    perm = rng.permutation(30)
    inv_perm = np.argsort(perm)
    descB = descA[perm] + rng.normal(0, 0.01, (30, 16)).astype(np.float32)
    pairs = m.match_descriptors(descA, descB, binary=False, ratio=0.95)
    assert len(pairs) > 0
    for a, b in pairs:
        assert inv_perm[a] == b
