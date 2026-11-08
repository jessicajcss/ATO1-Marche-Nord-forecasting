#!/usr/bin/env python3
"""Diebold–Mariano tests of every model against the seasonal-naïve benchmark (civil demand).

Reproduces the significance results reported in Section 3.2 of the article, using the
same test as the notebooks (`dm_test_hln` in ATO1_parsimonious_pipeline.ipynb: squared-error
loss, Harvey–Leybourne–Newbold small-sample correction), applied independently within each
of the five seeds. Positive DM = the model has larger squared errors than seasonal-naïve.

Input (withheld — contains observed values; see data/README_DATA.md):
    output_configc/configc_predictions_long.csv

Usage (from the repository root):
    python analysis/dm_tests_vs_seasonal_naive.py [--predictions PATH] [--out CSV]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parents[1]


def dm_test_hln(err1, err2, h=1):
    """Diebold–Mariano test with the Harvey–Leybourne–Newbold (1997) correction (copied from the notebooks)."""
    d = np.asarray(err1) ** 2 - np.asarray(err2) ** 2
    T_len = len(d)
    if T_len < 3:
        return np.nan, np.nan
    d_bar = d.mean()
    var_d = np.var(d, ddof=0)
    for lag in range(1, h):
        if lag >= T_len:
            break
        cov = np.cov(d[lag:], d[:-lag])[0, 1] if T_len - lag > 1 else 0.0
        var_d += 2 * (1 - lag / h) * cov
    var_d = max(var_d, 1e-12)
    dm_stat = d_bar / np.sqrt(var_d / T_len)
    correction = np.sqrt((T_len + 1 - 2 * h + h * (h - 1) / T_len) / T_len)
    dm_adj = dm_stat * correction
    p = float(2 * t_dist.sf(abs(dm_adj), df=T_len - 1))
    return float(dm_adj), p


def per_seed_dm(pred, experiment, target, horizon, model, ref='Seasonal naive', hac_h=None):
    # as in the notebooks' multi_seed_dm_test, the HAC truncation lag equals the forecast lead
    hac_h = horizon if hac_h is None else hac_h
    out = []
    q = pred[(pred.experiment == experiment) & (pred.target == target) & (pred.horizon == horizon)]
    for seed, g in q.groupby('seed'):
        a = g[g.model == model][['origin_idx', 'obs', 'pred']]
        b = g[g.model == ref][['origin_idx', 'obs', 'pred']]
        m = a.merge(b, on='origin_idx', suffixes=('_m', '_r')).dropna().sort_values('origin_idx')
        if len(m) < 5:
            continue
        dm, p = dm_test_hln((m.obs_m - m.pred_m).values, (m.obs_r - m.pred_r).values, h=hac_h)
        out.append(dict(seed=seed, n=len(m), DM=dm, p=p))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--predictions', type=Path, default=ROOT / 'output_configc' / 'configc_predictions_long.csv')
    ap.add_argument('--out', type=Path, default=None, help='optional CSV with the per-seed results')
    args = ap.parse_args(argv)
    if not args.predictions.exists():
        raise SystemExit(f'{args.predictions} not found (withheld; see data/README_DATA.md)')
    pred = pd.read_csv(args.predictions)

    cases = [('short_term_multivariate', 1, m) for m in
             ['XGBoost', 'LSTM', 'SARIMA', 'Persistence', 'Transformer', 'CNN-LSTM', 'MLP']]
    cases += [('long_term_multivariate', 5, 'XGBoost')]  # the single lead at which XGBoost exceeds the benchmark
    rows, per_seed = [], []
    for exp, h, m in cases:
        r = per_seed_dm(pred, exp, 'Fabbisogni', h, m)
        r.insert(0, 'model', m); r.insert(0, 'horizon', h); r.insert(0, 'experiment', exp)
        per_seed.append(r)
        rows.append(dict(experiment=exp, horizon=h, model=m, n_seeds=len(r), n_origins=int(r.n.median()),
                         median_DM=r.DM.median(), median_p=r.p.median(),
                         seeds_p_lt_0_05=int((r.p < 0.05).sum()), seeds_p_lt_0_01=int((r.p < 0.01).sum())))
    summary = pd.DataFrame(rows)
    with pd.option_context('display.width', 140, 'display.float_format', '{:.3f}'.format):
        print('Civil demand (Fabbisogni): each model vs seasonal-naive, HLN-corrected DM test per seed\n')
        print(summary.to_string(index=False))
    if args.out:
        pd.concat(per_seed).to_csv(args.out, index=False)
        print(f'\nper-seed results written to {args.out}')


if __name__ == '__main__':
    main()
