"""
stats.py — the inferential layer.

Four analyses, each answering a question the raw tables cannot:

  1. FAILURE PREDICTION.  Given only quantities observable WITHOUT ground truth,
     can we tell that a converged registration is actually wrong? This is the
     operational question — on real PDS data there is no truth to check against.
     Measured on 159 converged runs of which 25 are silently wrong.

  2. DOSE-RESPONSE.  Fit success probability against solar azimuth difference
     per method and report the LD50 — the azimuth difference at which a method
     is a coin flip. This turns "SIFT stops working around 30-45°" into a number
     with a confidence interval.

  3. VARIANCE DECOMPOSITION.  Of the spread in registration error, how much is
     attributable to the method, to the illumination condition, and to which
     patch of terrain you happened to draw? If terrain dominates, comparing
     methods on a single scene is meaningless.

  4. PAIRED COMPARISON.  Methods are run on identical terrain-condition pairs,
     so differences should be tested paired, not as independent samples. A
     paired bootstrap respects that and gives an interval on the difference.

Small-sample discipline
-----------------------
25 positives is not many. A gradient-boosted model on 17 features will memorise
them, so every number here is cross-validated, and reported two ways: a random
stratified split (optimistic — related runs can straddle the fold boundary) and
a split grouped by terrain seed (honest — the model must generalise to unseen
ground). Where the two disagree, the grouped figure is the one to believe.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

CSVS = ['E1_illumination', 'E2_scale', 'E3_ablation', 'E4_noise']

# Everything here is computable on real data with no reference truth.
OBSERVABLE = [
    'inlier_ratio', 'n_final', 'n_putative', 'n_inliers_raw', 'rmse_reproj',
    'resid_moran_x', 'resid_moran_y', 'resid_bias', 'resid_aniso',
    'unif_after_coverage', 'unif_after_entropy', 'unif_after_clark_evans',
    'unif_after_ripley_dev', 'unif_after_uniformity', 't_total',
    'rmse_ci_width', 'putative_to_final',
]

PRETTY = {
    'inlier_ratio': 'Inlier ratio',
    'n_final': 'Final tie points',
    'n_putative': 'Putative matches',
    'n_inliers_raw': 'Raw inliers',
    'rmse_reproj': 'Reprojection RMSE',
    'resid_moran_x': "Moran's I, x residuals",
    'resid_moran_y': "Moran's I, y residuals",
    'resid_bias': 'Residual mean bias',
    'resid_aniso': 'Residual anisotropy',
    'unif_after_coverage': 'Grid coverage',
    'unif_after_entropy': 'Cell entropy',
    'unif_after_clark_evans': 'Clark–Evans index',
    'unif_after_ripley_dev': 'Ripley K deviation',
    'unif_after_uniformity': 'Composite uniformity',
    't_total': 'Runtime',
    'rmse_ci_width': 'Bootstrap CI width',
    'putative_to_final': 'Survival fraction',
}


def load(out_dir='outputs') -> pd.DataFrame:
    frames = []
    for name in CSVS:
        try:
            frames.append(pd.read_csv(f'{out_dir}/{name}.csv'))
        except FileNotFoundError:
            continue
    df = pd.concat(frames, ignore_index=True)
    df['rmse_ci_width'] = df['rmse_ci_hi'] - df['rmse_ci_lo']
    df['putative_to_final'] = df['n_final'] / df['n_putative'].replace(0, np.nan)
    return df


# --------------------------------------------------------------------------
# 1. failure prediction
# --------------------------------------------------------------------------

def failure_dataset(df: pd.DataFrame, threshold: float = 1.0):
    """Converged runs only — the population where a silent failure is possible."""
    d = df[df['ok'].astype(bool) & df['rmse_true'].notna()].copy()
    y = (d['rmse_true'] > threshold).astype(int).to_numpy()
    X = d[OBSERVABLE].to_numpy(float)
    groups = d['seed'].fillna(-1).to_numpy()
    return X, y, groups, d


def _cv_scores(X, y, groups, model_fn, mode='stratified', n_splits=5, seed=0):
    """Out-of-fold probabilities, so every score is on data the model never saw."""
    oof = np.full(len(y), np.nan)
    if mode == 'grouped':
        uniq = np.unique(groups)
        n_splits = int(min(n_splits, len(uniq)))
        if n_splits < 2:
            return oof
        splitter = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed).split(X, y)
    for tr, te in splitter:
        if len(np.unique(y[tr])) < 2:
            continue
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def _auc_ci(y, p, n_boot=2000, seed=0):
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return None, None, None
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, p)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        bs.append(roc_auc_score(y[i], p[i]))
    return float(base), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _logreg():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=0.4, max_iter=3000,
                                            class_weight='balanced'))


def _gbm():
    return make_pipeline(SimpleImputer(strategy='median'),
                         GradientBoostingClassifier(n_estimators=120, max_depth=2,
                                                    learning_rate=0.06,
                                                    subsample=0.85, random_state=0))


def failure_analysis(df: pd.DataFrame, threshold: float = 1.0) -> dict:
    X, y, groups, d = failure_dataset(df, threshold)
    out = dict(n=int(len(y)), n_fail=int(y.sum()),
               base_rate=float(y.mean()), threshold=threshold)

    # A confound to rule out before believing any of this: silent failures are
    # concentrated in particular configurations (no-scale-prior runs, the sparse
    # matchers), and count features like "number of putative matches" are strong
    # signatures of WHICH method ran. A model could score well by recognising
    # the configuration rather than by detecting error. Splitting on
    # configuration forces it to generalise to setups it has never seen.
    config = (d['method'].astype(str) + '|' + d['variant'].fillna('-').astype(str) +
              '|' + d['src_sensor'].fillna('-').astype(str) +
              d['ref_sensor'].fillna('-').astype(str)).to_numpy().astype(str)

    models = {'logistic': _logreg, 'gbm': _gbm}
    out['models'] = {}
    best = None
    for name, fn in models.items():
        entry = {}
        for mode in ('stratified', 'grouped', 'config'):
            g = config if mode == 'config' else groups
            p = _cv_scores(X, y, g, fn, mode='grouped' if mode != 'stratified' else 'stratified')
            a, lo, hi = _auc_ci(y, p)
            entry[mode] = dict(auc=a, lo=lo, hi=hi)
            # Operating curves are reported from the CONFIG split — the only one
            # that forces generalisation to unseen setups. Measured: the gradient
            # boosting model scores 0.990 on a random split and 0.467 (chance) on
            # this one, i.e. it was recognising configurations, not detecting
            # error. The regularised linear model holds at 0.890, so that is the
            # model that gets deployed.
            if mode == 'config' and a is not None:
                if best is None or a > best[1]:
                    best = (name, a, p)
        out['models'][name] = entry

    # single-feature AUCs: which diagnostics carry signal on their own?
    singles = []
    for j, f in enumerate(OBSERVABLE):
        v = X[:, j]
        ok = np.isfinite(v)
        if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
            continue
        a = roc_auc_score(y[ok], v[ok])
        singles.append(dict(feature=f, label=PRETTY.get(f, f),
                            auc=float(max(a, 1 - a)),
                            direction='higher→failure' if a > 0.5 else 'lower→failure',
                            rho=float(sps.spearmanr(v[ok], y[ok]).statistic)))
    singles.sort(key=lambda r: -r['auc'])
    out['single_feature'] = singles

    # Hardest test available: restrict to the dense pipeline alone, so method
    # identity carries no information and the model must judge runs of the SAME
    # configuration against each other.
    dm = (d['method'] == 'dense').to_numpy()
    if dm.sum() > 30 and 0 < y[dm].sum() < dm.sum():
        Xd, yd, gd = X[dm], y[dm], groups[dm]
        pd_ = _cv_scores(Xd, yd, gd, _logreg, mode='grouped')
        a, lo, hi = _auc_ci(yd, pd_)
        out['within_dense'] = dict(n=int(dm.sum()), n_fail=int(yd.sum()),
                                   auc=a, lo=lo, hi=hi)
        cfg_d = config[dm]
        pc_ = _cv_scores(Xd, yd, cfg_d, _logreg, mode='grouped')
        a2, lo2, hi2 = _auc_ci(yd, pc_)
        out['within_dense']['auc_config'] = a2
        out['within_dense']['lo_config'] = lo2
        out['within_dense']['hi_config'] = hi2
    else:
        out['within_dense'] = None

    # operating curve of the best cross-validated model
    if best is not None:
        name, auc, p = best
        ok = np.isfinite(p)
        yv, pv = y[ok], p[ok]
        fpr, tpr, thr = roc_curve(yv, pv)
        step = max(1, len(fpr) // 60)
        out['roc'] = dict(model=name,
                          fpr=[float(v) for v in fpr[::step]],
                          tpr=[float(v) for v in tpr[::step]])
        # full sweep so the dashboard can move the threshold live
        grid = np.linspace(0.02, 0.98, 49)
        sweep = []
        for t in grid:
            pred = (pv >= t).astype(int)
            tp = int(((pred == 1) & (yv == 1)).sum())
            fp = int(((pred == 1) & (yv == 0)).sum())
            fn = int(((pred == 0) & (yv == 1)).sum())
            tn = int(((pred == 0) & (yv == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else None
            rec = tp / (tp + fn) if tp + fn else None
            f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else 0.0
            sweep.append(dict(t=float(t), tp=tp, fp=fp, fn=fn, tn=tn,
                              precision=prec, recall=rec, f1=float(f1)))
        out['sweep'] = sweep
        out['scores'] = [dict(p=float(a), y=int(b)) for a, b in zip(pv, yv)]

        # calibration in quantile bins
        try:
            qs = np.quantile(pv, np.linspace(0, 1, 6))
            qs = np.unique(qs)
            cal = []
            for a, b in zip(qs[:-1], qs[1:]):
                m = (pv >= a) & (pv <= b)
                if m.sum() >= 5:
                    cal.append(dict(pred=float(pv[m].mean()),
                                    obs=float(yv[m].mean()), n=int(m.sum())))
            out['calibration'] = cal
        except Exception:
            out['calibration'] = []
    return out


# --------------------------------------------------------------------------
# 2. dose-response
# --------------------------------------------------------------------------

def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(k * (x - x0)))


def dose_response(df: pd.DataFrame, n_boot: int = 600) -> list:
    """Success probability vs solar azimuth difference, per method, with LD50."""
    d = df[(df['experiment'] == 'E1_illumination') & df['d_az'].notna()]
    out = []
    for method, g in d.groupby('method'):
        x = g['d_az'].to_numpy(float)
        y = g['ok'].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            out.append(dict(method=method, ld50=None, lo=None, hi=None,
                            points=_binned(x, y), always=bool(y.all())))
            continue
        try:
            popt, _ = optimize.curve_fit(_logistic, x, y, p0=[60.0, 0.08],
                                         maxfev=20000)
        except Exception:
            popt = [np.nan, np.nan]
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(x), len(x))
            if len(np.unique(y[i])) < 2:
                continue
            try:
                p, _ = optimize.curve_fit(_logistic, x[i], y[i], p0=popt if
                                          np.isfinite(popt[0]) else [60.0, 0.08],
                                          maxfev=8000)
                if 0 <= p[0] <= 360:
                    bs.append(p[0])
            except Exception:
                continue
        curve = [dict(x=float(v), y=float(_logistic(v, *popt)))
                 for v in np.linspace(0, 180, 61)] if np.isfinite(popt[0]) else []
        out.append(dict(
            method=method,
            ld50=float(popt[0]) if np.isfinite(popt[0]) else None,
            lo=float(np.percentile(bs, 2.5)) if len(bs) > 30 else None,
            hi=float(np.percentile(bs, 97.5)) if len(bs) > 30 else None,
            curve=curve, points=_binned(x, y), always=bool(y.all())))
    return out


def _binned(x, y):
    out = []
    for v in np.unique(x):
        m = x == v
        k, n = int(y[m].sum()), int(m.sum())
        # Wilson interval — correct for proportions near 0 and 1, unlike normal
        lo, hi = _wilson(k, n)
        out.append(dict(x=float(v), p=k / n, n=n, lo=lo, hi=hi))
    return out


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (float(max(0, ctr - half)), float(min(1, ctr + half)))


# --------------------------------------------------------------------------
# 3. variance decomposition
# --------------------------------------------------------------------------

def variance_decomposition(df: pd.DataFrame) -> dict:
    """
    How much of the spread in log error is method, condition, terrain, residual?

    Errors span orders of magnitude, so the decomposition runs on log10(error);
    on the raw scale a single 400 px outlier would swamp every other term.
    """
    d = df[(df['experiment'] == 'E1_illumination') & df['ok'].astype(bool) &
           df['rmse_true'].notna()].copy()
    d['le'] = np.log10(d['rmse_true'].clip(lower=1e-3))
    grand = d['le'].mean()
    total = ((d['le'] - grand) ** 2).sum()
    if total <= 0:
        return dict(components=[], n=int(len(d)))

    comps = []
    for name, col in [('Method', 'method'), ('Illumination Δaz', 'd_az'),
                      ('Terrain draw', 'seed')]:
        ss = sum(len(g) * (g['le'].mean() - grand) ** 2 for _, g in d.groupby(col))
        comps.append(dict(name=name, share=float(ss / total)))
    explained = sum(c['share'] for c in comps)
    comps.append(dict(name='Unexplained', share=float(max(0.0, 1 - explained))))
    return dict(components=comps, n=int(len(d)),
                sd_log10=float(d['le'].std()))


# --------------------------------------------------------------------------
# 4. paired comparison
# --------------------------------------------------------------------------

def paired_comparison(df: pd.DataFrame, n_boot: int = 4000) -> list:
    """
    Dense against each baseline on identical (terrain, illumination) cells.

    Unpaired tests would attribute to the method a difference that is really the
    luck of the terrain draw, which the variance decomposition shows is large.
    """
    d = df[df['experiment'] == 'E1_illumination'].copy()
    piv = d.pivot_table(index=['seed', 'd_az'], columns='method',
                        values='ok', aggfunc='max')
    out = []
    if 'dense' not in piv.columns:
        return out
    rng = np.random.default_rng(0)
    for m in piv.columns:
        if m == 'dense':
            continue
        sub = piv[['dense', m]].dropna()
        if len(sub) < 5:
            continue
        a = sub['dense'].to_numpy(float)
        b = sub[m].to_numpy(float)
        diff = a - b
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean()
              for _ in range(n_boot)]
        # McNemar on discordant cells
        n01 = int(((a == 0) & (b == 1)).sum())
        n10 = int(((a == 1) & (b == 0)).sum())
        pval = float(sps.binomtest(n10, n10 + n01, 0.5).pvalue) if (n10 + n01) else 1.0
        out.append(dict(method=m, n_pairs=int(len(sub)),
                        delta=float(diff.mean()),
                        lo=float(np.percentile(bs, 2.5)),
                        hi=float(np.percentile(bs, 97.5)),
                        wins=n10, losses=n01, p=pval))
    out.sort(key=lambda r: -r['delta'])
    return out


# --------------------------------------------------------------------------

def run_explorer_rows(df: pd.DataFrame, max_rows: int = 900) -> list:
    d = df[df['ok'].astype(bool) & df['rmse_true'].notna()].copy()
    cols = OBSERVABLE + ['rmse_true', 'method', 'd_az', 'experiment']
    d = d[[c for c in cols if c in d.columns]]
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=0)
    recs = []
    for _, r in d.iterrows():
        rec = {}
        for c in d.columns:
            v = r[c]
            if isinstance(v, (int, float, np.floating, np.integer)):
                v = float(v)
                rec[c] = None if not np.isfinite(v) else v
            else:
                rec[c] = v
        recs.append(rec)
    return recs


def main(out_dir='outputs'):
    df = load(out_dir)
    payload = dict(
        failure=failure_analysis(df),
        dose=dose_response(df),
        variance=variance_decomposition(df),
        paired=paired_comparison(df),
        runs=run_explorer_rows(df),
        features=[dict(key=k, label=PRETTY[k]) for k in OBSERVABLE],
        n_total=int(len(df)),
    )
    with open(f'{out_dir}/stats.json', 'w') as f:
        json.dump(payload, f, allow_nan=False)
    return payload


if __name__ == '__main__':
    p = main()
    f = p['failure']
    print(f"failure: n={f['n']} positives={f['n_fail']} base={f['base_rate']:.3f}")
    for k, v in f['models'].items():
        for mode, s in v.items():
            if s['auc'] is not None:
                print(f"  {k:9s} {mode:11s} AUC={s['auc']:.3f} "
                      f"[{s['lo']:.3f}, {s['hi']:.3f}]")
    print('top single features:')
    for r in f['single_feature'][:6]:
        print(f"  {r['label']:26s} AUC={r['auc']:.3f}  {r['direction']}")
    print('dose-response LD50:')
    for r in p['dose']:
        if r['ld50'] is not None:
            print(f"  {r['method']:8s} {r['ld50']:6.1f}° "
                  f"[{r['lo']:.0f}, {r['hi']:.0f}]" if r['lo'] else
                  f"  {r['method']:8s} {r['ld50']:6.1f}°")
        else:
            print(f"  {r['method']:8s} never fell below 50% success")
    print('variance shares:', {c['name']: round(c['share'], 3)
                               for c in p['variance']['components']})
    print('paired vs dense:')
    for r in p['paired']:
        print(f"  {r['method']:8s} Δsuccess={r['delta']:+.3f} "
              f"[{r['lo']:+.3f}, {r['hi']:+.3f}] wins={r['wins']} losses={r['losses']} p={r['p']:.2e}")
