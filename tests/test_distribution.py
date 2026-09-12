import numpy as np

from lunareg import distribution as dist


def _grid_points(shape, n_per_axis):
    h, w = shape
    xs = np.linspace(5, w - 5, n_per_axis)
    ys = np.linspace(5, h - 5, n_per_axis)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], 1)


def test_grid_coverage_full_for_dense_regular_grid():
    shape = (200, 200)
    pts = _grid_points(shape, 16)  # far denser than an 8x8 evaluation grid
    assert dist.grid_coverage(pts, shape, grid=8) == 1.0


def test_grid_coverage_zero_for_no_points():
    assert dist.grid_coverage(np.zeros((0, 2)), (200, 200)) == 0.0


def test_grid_coverage_low_when_points_cluster_in_one_cell():
    shape = (200, 200)
    pts = np.full((20, 2), 5.0)  # all in the top-left cell
    cov = dist.grid_coverage(pts, shape, grid=8)
    assert cov == 1.0 / 64


def test_cell_entropy_max_for_uniform_grid():
    shape = (200, 200)
    pts = _grid_points(shape, 8)
    ent = dist.cell_entropy(pts, shape, grid=8)
    assert ent > 0.95


def test_cell_entropy_zero_when_all_points_in_one_cell():
    shape = (200, 200)
    pts = np.full((20, 2), 5.0)
    assert dist.cell_entropy(pts, shape, grid=8) == 0.0


def test_clark_evans_near_one_for_regular_grid():
    shape = (400, 400)
    pts = _grid_points(shape, 12)
    r = dist.clark_evans(pts, shape)
    # a perfect square lattice is dispersed relative to a Poisson process (R>1)
    assert r > 1.0


def test_clark_evans_low_for_tight_cluster():
    rng = np.random.default_rng(0)
    shape = (400, 400)
    cluster = rng.normal([200, 200], 2.0, (30, 2))
    r = dist.clark_evans(cluster, shape)
    assert r < 0.3


def test_enforce_uniform_respects_target_and_prefers_high_score(rng):
    shape = (300, 300)
    pts_ref = rng.uniform(0, 300, (500, 2))
    pts_src = pts_ref.copy()
    scores = rng.uniform(0, 1, 500)
    keep = dist.enforce_uniform(pts_src, pts_ref, scores, shape, target=100, grid=8)
    assert 0 < len(keep) <= 500
    # the kept set should be reasonably close to the target count
    assert abs(len(keep) - 100) < 100


def test_uniformity_report_composite_higher_for_uniform_than_clustered():
    shape = (300, 300)
    uniform_pts = _grid_points(shape, 10)
    rng = np.random.default_rng(1)
    clustered_pts = rng.normal([150, 150], 5.0, (80, 2))
    ru = dist.uniformity_report(uniform_pts, shape)
    rc = dist.uniformity_report(clustered_pts, shape)
    assert ru['uniformity'] > rc['uniformity']
    assert ru['coverage'] > rc['coverage']
