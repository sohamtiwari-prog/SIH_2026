import numpy as np

from lunareg import metrics as mt


def _random_homography(rng, scale=1.2, rot_deg=8.0, tx=15.0, ty=-7.0):
    th = np.deg2rad(rot_deg)
    c, s = np.cos(th), np.sin(th)
    H = np.array([[scale * c, -scale * s, tx],
                 [scale * s, scale * c, ty],
                 [0.0, 0.0, 1.0]])
    return H


def test_apply_H_identity(rng):
    pts = rng.uniform(0, 100, (10, 2))
    out = mt.apply_H(np.eye(3), pts)
    assert np.allclose(out, pts)


def test_apply_H_matches_manual_projective(rng):
    H = _random_homography(rng)
    H[2, 0], H[2, 1] = 1e-4, -2e-4  # make it genuinely projective
    pts = rng.uniform(0, 50, (5, 2))
    got = mt.apply_H(H, pts)
    for (x, y), (gx, gy) in zip(pts, got):
        v = H @ np.array([x, y, 1.0])
        assert np.allclose((gx, gy), (v[0] / v[2], v[1] / v[2]))


def test_residuals_zero_for_exact_correspondences(rng):
    H = _random_homography(rng)
    ptsA = rng.uniform(0, 100, (20, 2))
    ptsB = mt.apply_H(H, ptsA)
    res = mt.residuals(H, ptsA, ptsB)
    assert mt.rmse(res) < 1e-9


def test_rmse_empty_is_nan():
    assert np.isnan(mt.rmse(np.zeros((0, 2))))


def test_true_geometric_error_zero_for_identical_homography(rng):
    H = _random_homography(rng)
    e = mt.true_geometric_error(H, H, (200, 200))
    assert e['rmse_true'] < 1e-9
    assert e['max_true'] < 1e-9


def test_true_geometric_error_detects_translation_bias():
    H_true = np.eye(3)
    H_est = np.array([[1, 0, 3.0], [0, 1, -2.0], [0, 0, 1.0]])
    e = mt.true_geometric_error(H_est, H_true, (100, 100))
    assert np.isclose(e['bias_x'], 3.0, atol=1e-6)
    assert np.isclose(e['bias_y'], -2.0, atol=1e-6)
    assert np.isclose(e['rmse_true'], np.hypot(3.0, 2.0), atol=1e-6)


def test_match_quality_counts_correct_matches_within_tolerance(rng):
    H_true = np.eye(3)
    ptsA = rng.uniform(0, 100, (30, 2))
    ptsB = ptsA.copy()
    ptsB[:10] += 20.0  # push 10 points outside tolerance
    q = mt.match_quality(ptsA, ptsB, H_true, tol=3.0)
    assert q['ncm'] == 20
    assert np.isclose(q['cmr'], 20 / 30)
    assert q['n_put'] == 30


def test_morans_I_high_for_spatially_smooth_field(rng):
    pts = rng.uniform(0, 100, (60, 2))
    smooth = pts[:, 0] + rng.normal(0, 0.01, 60)  # a spatial gradient: clusters
    noise = rng.normal(0, 1, 60)                  # spatially random
    i_smooth = mt.morans_I(pts, smooth)
    i_noise = mt.morans_I(pts, noise)
    assert i_smooth > i_noise


def test_rmse_bootstrap_ci_brackets_point_estimate(rng):
    res = rng.normal(0, 1.0, (200, 2))
    r, lo, hi = mt.rmse_bootstrap_ci(res, n_boot=500, seed=1)
    assert lo <= r <= hi


def test_loo_rmse_near_zero_for_noiseless_affine_fit(rng):
    H = _random_homography(rng)
    ptsA = rng.uniform(0, 100, (40, 2))
    ptsB = mt.apply_H(H, ptsA)

    def affine_fit(a, b):
        A = np.hstack([a, np.ones((len(a), 1))])
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        M = np.vstack([sol.T, [0, 0, 1.0]])
        return M

    err = mt.loo_rmse(ptsA, ptsB, affine_fit, max_pts=40, seed=0)
    assert err < 1e-6
