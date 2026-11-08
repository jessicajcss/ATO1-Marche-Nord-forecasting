#!/usr/bin/env python3
"""Regenerate the manuscript figures (Figs. 2-13) from the repository outputs.

Usage (from the repository root):

    python figures/make_figures.py                 # all figures whose inputs are available
    python figures/make_figures.py --only 5 6 13   # a subset
    python figures/make_figures.py --dpi 300 --out figures/output

Inputs
------
Public (shipped in this repository):
    output_configc/master_metrics_configc_agg.csv       Figs. 5, 6, 7*, 8*, 10, 12, 13
    output_v2/master_metrics_v2_agg.csv                 Fig. 12
    output_part2_5seed/master_metrics_part2_agg_final.csv  Fig. 12
Withheld (operator data; available on request, see data/README_DATA.md):
    data/Dataset.xlsx                                   Figs. 2, 3, 9, 11
    output_configc/configc_predictions_long.csv         Figs. 7, 8, 9, 11
Fig. 4 is a schematic and needs no data. Fig. 1 (study-area map) was produced in GIS
and is not generated here. Figures whose inputs are missing are skipped with a message.

Each figure is written as a PNG (at --dpi, default 300) and as a vector PDF with
embedded TrueType fonts. By default the figures are drawn as submitted with the
article, i.e. without figure-level titles and footnotes (that information is in the
captions); pass --titles to draw them as well.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]

# Okabe-Ito / Wong colour-blind-safe palette
VERM, BLUE, GREEN, PINK, RED = '#d55e00', '#0072b2', '#009e73', '#cc79a7', '#c0392b'
GBAND, RBAND = '#f4f9f6', '#faf6f5'
SHOW_TITLES = False  # set by --titles: draw figure-level titles / footnotes (captions carry them in the article)


# ---------------------------------------------------------------- helpers
def base_rc():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titleweight': 'bold',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})


def grey(a, col='#666666'):
    for sp in a.spines.values():
        sp.set_color(col)
    a.tick_params(color=col)


def nse(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float)
    return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)


class Inputs:
    """Lazily loaded inputs; raises FileNotFoundError when a withheld file is absent."""

    def __init__(self, data_path, pred_path):
        self.data_path, self.pred_path = Path(data_path), Path(pred_path)
        self._cache = {}

    def _csv(self, rel):
        if rel not in self._cache:
            self._cache[rel] = pd.read_csv(ROOT / rel)
        return self._cache[rel]

    @property
    def cfg(self):
        return self._csv('output_configc/master_metrics_configc_agg.csv')

    @property
    def v2(self):
        return self._csv('output_v2/master_metrics_v2_agg.csv')

    @property
    def p2(self):
        return self._csv('output_part2_5seed/master_metrics_part2_agg_final.csv')

    @property
    def dataset(self):
        if 'ds' not in self._cache:
            if not self.data_path.exists():
                raise FileNotFoundError(self.data_path)
            df = pd.read_excel(self.data_path)
            df['Data'] = pd.to_datetime(df['Data'])
            # network losses are the balance residual (the Perdite column is left empty in the source file)
            df['Perdite'] = df.Sorgenti + df.Pozzi + df.Potabilizzatori - df.Fabbisogni
            self._cache['ds'] = df.sort_values('Data').reset_index(drop=True)
        return self._cache['ds']

    @property
    def pred(self):
        if 'pred' not in self._cache:
            if not self.pred_path.exists():
                raise FileNotFoundError(self.pred_path)
            d = pd.read_csv(self.pred_path, parse_dates=['target_date'])
            self._cache['pred'] = d[d.experiment.isin(['short_term_multivariate', 'long_term_multivariate']) & (d.horizon == 1)]
        return self._cache['pred']


def hide_figure_titles(fig):
    """Hide figure-level titles and footnotes after the layout was computed with them, so the
    saved figure equals the titled one with those elements cropped off (as submitted)."""
    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    for t in fig.texts:
        t.set_visible(False)
    if len(fig.axes) == 1:  # single-panel figures: the axes title is the figure title
        a = fig.axes[0]
        for t in (a.title, a._left_title, a._right_title):
            t.set_visible(False)


def save(fig, out, name, dpi):
    out.mkdir(parents=True, exist_ok=True)
    if not SHOW_TITLES:
        hide_figure_titles(fig)
    fig.savefig(out / f'{name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out / f'{name}.png'


# ---------------------------------------------------------------- Fig 2
def fig02(I, out, dpi):
    base_rc(); df = I.dataset; M = 1e6
    fig, ax = plt.subplots(2, 1, figsize=(8.5, 5.3), sharex=True)
    fig.suptitle('Raw ATO1 monthly volumes', fontsize=11.5, fontweight='bold')
    for a in ax:
        a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', alpha=0.25); a.set_ylabel('million m$^3$/month')
    ax[0].plot(df.Data, df.Fabbisogni / M, '-o', color=VERM, lw=2, ms=3.5, label='Demand (Fabbisogni)')
    ax[0].plot(df.Data, df.Perdite / M, '-s', color=RED, lw=2, ms=3.5, label='Losses (Perdite)')
    ax[0].set_title('The two forecast targets, 2017-2021', loc='left', fontsize=10)
    ax[1].plot(df.Data, df.Sorgenti / M, '-^', color=BLUE, lw=2, ms=3.5, label='Springs (Sorgenti)')
    ax[1].plot(df.Data, df.Pozzi / M, '-D', color=GREEN, lw=2, ms=3.5, label='Wells (Pozzi)')
    ax[1].set_title('Two of the supply sources', loc='left', fontsize=10)
    for a in ax:
        a.legend(loc='lower right', ncol=2, frameon=False, fontsize=9); grey(a)
    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=1.4)
    return save(fig, out, 'Figure02_raw_monthly_volumes', dpi)


# ---------------------------------------------------------------- Fig 3
def fig03(I, out, dpi):
    base_rc(); df = I.dataset.copy(); M = 1e6; df['m'] = df.Data.dt.month
    fig, ax = plt.subplots(1, 2, figsize=(8.9, 3.6))
    fig.suptitle('Month-of-year profile: demand is tight (predictable), losses are diffuse', fontsize=10.5, fontweight='bold')
    for a, (col, ttl, c) in zip(ax, [('Fabbisogni', 'Demand by calendar month', VERM), ('Perdite', 'Losses by calendar month', RED)]):
        data = [df[df.m == k][col].values / M for k in range(1, 13)]
        bp = a.boxplot(data, patch_artist=True, widths=0.55,
                       medianprops=dict(color='#1a1a1a', lw=1.6), whiskerprops=dict(color='black', lw=1),
                       capprops=dict(color='black', lw=1), flierprops=dict(marker='o', ms=1.8, mfc='black', mec='black'))
        for b in bp['boxes']:
            b.set_facecolor((*matplotlib.colors.to_rgb(c), 0.55)); b.set_edgecolor((*matplotlib.colors.to_rgb(c), 0.8))
        a.set_xticks([1, 4, 7, 10]); a.set_xticklabels(['J', 'A', 'J', 'O']); a.set_xlabel('month (Jan..Dec)')
        a.set_title(ttl, fontsize=10); a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', alpha=0.25)
    ax[0].set_ylabel('million m$^3$/month')
    for a in ax:
        grey(a)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, out, 'Figure03_month_of_year_profile', dpi)


# ---------------------------------------------------------------- Fig 4 (schematic)
def fig04(I, out, dpi):
    base_rc()
    TEAL, SUB, EDGE, FILL, ARR, LAB = '#123f4d', '#2a6b7c', '#9fb3bd', '#eaf1f4', '#333333', '#2b2b2b'

    def panel(ax, letter, subtitle, inputs):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
        bx0, bx1, by0, by1 = 3.55, 6.75, 2.2, 7.8
        ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0, boxstyle='round,pad=0.08,rounding_size=0.25',
                                    fc=FILL, ec=EDGE, lw=1.4))
        ax.text((bx0 + bx1) / 2, (by0 + by1) / 2, 'Water-resources\ndemand & loss\nforecasting model',
                ha='center', va='center', fontsize=9.6, fontweight='bold', color=TEAL)
        for y, lab in zip(np.linspace(by1 - 0.45, by0 + 0.45, len(inputs)), inputs):
            ax.annotate('', xy=(bx0 - 0.02, y), xytext=(3.2, y), arrowprops=dict(arrowstyle='-|>', color=ARR, lw=1.4))
            ax.text(3.18, y, lab, ha='right', va='center', fontsize=8.2, color=LAB)
        ax.text(3.18, by1 + 0.32, 'monthly inputs, months k, k−1, …, k−m', ha='right', va='center', fontsize=7.6,
                style='italic', color=SUB)
        for y, lab in zip([by1 - 1.45, by0 + 1.45], ['CWD for\nmonths k+1 … k+n', 'NL for\nmonths k+1 … k+n']):
            ax.annotate('', xy=(7.85, y), xytext=(bx1 + 0.02, y), arrowprops=dict(arrowstyle='-|>', color=ARR, lw=1.6))
            ax.text(8.02, y, lab, ha='left', va='center', fontsize=8.2, color=LAB)
        ax.text(0.6, 9.45, letter, fontsize=16, fontweight='bold', color='#111111', ha='center', va='center')
        ax.text(5.0, 1.2, subtitle, ha='center', va='center', fontsize=9.6, style='italic', color=SUB)

    # inputs of the parsimonious specification (Table 2); the two schemes are the *_multivariate and
    # *_autoregressive experiments of the pipelines
    base = ['Annual population (Popolazione)', 'Spring abstraction (Sorgenti)', 'Well-field abstraction (Pozzi)',
            'Treated drinking water (Potabilizzatori)', 'Calendar terms (month sin/cos, year index)',
            'Seismic covariates (step flag, decay)']
    fig, ax = plt.subplots(2, 1, figsize=(5.0, 6.73))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.08)
    panel(ax[0], 'A', 'A · Multivariate scheme — drivers and covariates', base)
    panel(ax[1], 'B', 'B · Autoregressive scheme — adds one-month lags of both targets',
          base + ['CWD, lagged one month', 'NL, lagged one month'])
    return save(fig, out, 'Figure04_forecast_design', dpi)


# ---------------------------------------------------------------- Fig 5 / 6 / 10 helpers
LINES = {  # label, colour, linestyle, marker, linewidth
    'Seasonal naive': ('Seasonal-naive (benchmark)', '#111111', '-', 'o', 3.0),
    'Persistence': ('Persistence', '#8a8a8a', '--', 's', 2.2),
    'SARIMA': ('SARIMA', BLUE, '-', '^', 2.4),
    'XGBoost': ('XGBoost', GREEN, '-', 'D', 2.4),
    'LSTM': ('LSTM', VERM, '-', 'o', 2.4),
    'Transformer': ('Transformer', PINK, '-', 'v', 2.4)}


def long_series(I, target, model):
    q = I.cfg[(I.cfg.experiment == 'long_term_multivariate') & (I.cfg.target == target) & (I.cfg.model == model)].sort_values('horizon')
    return q.horizon.values, q.NSE_mean.values


def fig05(I, out, dpi):
    base_rc()
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 5.62))
    fig.suptitle('Forecast skill by horizon - parsimonious (9-feature) specification', fontsize=12, fontweight='bold', y=0.985)
    fig.text(0.5, 0.925, 'ATO1 Marche Nord - mean of 5 seeds - leakage-free rolling-origin', ha='center', fontsize=8.5, color='#555555')
    floor = -1.58
    for a, tgt, ttl, top in [(ax[0], 'Fabbisogni', 'A   Civil water demand', 1.0), (ax[1], 'Perdite', 'B   Network losses', 0.4)]:
        a.axhspan(0, top, color=GBAND, zorder=0); a.axhspan(-1.6, 0, color=RBAND, zorder=0)
        a.axhline(0, color='#444444', lw=1.2, zorder=1)
        off = []
        for m, (lab, c, ls, mk, lw) in LINES.items():
            h, v = long_series(I, tgt, m)
            if v.min() < -1.6:
                off.append(f'{m}->{v.min():.1f}')
            a.plot(h, np.maximum(v, floor), ls=ls, color=c, lw=lw, marker=mk, ms=5.5 if mk != 'D' else 5.0,
                   mec='white', mew=0.8, label=lab, zorder=3)
        if off:
            a.text(6.5, -1.535, 'off-axis: ' + '  '.join(off), ha='center', va='bottom', fontsize=7.6, color='#9c4a38')
        a.set_ylim(-1.6, top); a.set_xlim(1, 12); a.set_xticks(range(1, 13))
        a.set_title(ttl, loc='left', fontsize=11); a.set_xlabel('horizon (months ahead)', fontsize=10)
        a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', alpha=0.3); grey(a); a.tick_params(labelsize=10)
    ax[0].set_ylabel('Nash-Sutcliffe Efficiency (NSE)', fontsize=10)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.875), ncol=6, frameon=False, fontsize=8.5, handlelength=2.6)
    # footnote derived from the data rather than hard-coded
    w = I.cfg[(I.cfg.experiment == 'long_term_multivariate') & (I.cfg.target == 'Fabbisogni')].pivot(
        index='horizon', columns='model', values='NSE_mean')
    trained = [m for m in w.columns if m not in ('Seasonal naive', 'Persistence')]
    beat = (w[trained].max(axis=1) - w['Seasonal naive'])
    beat = beat[beat > 0]
    if len(beat):
        parts = [f"h={h} ({w.loc[h, trained].idxmax()}, +{d:.2f})" for h, d in beat.items()]
        note = 'Seasonal-naive beaten for demand only at ' + ', '.join(parts)
    else:
        note = 'Seasonal-naive unbeaten for demand at every horizon'
    allneg = (I.cfg[(I.cfg.experiment == 'long_term_multivariate') & (I.cfg.target == 'Perdite')].NSE_mean < 0).all()
    note += '; every model <0 for losses.' if allneg else '.'
    fig.text(0.5, 0.012, 'NSE=0 = historical mean. ' + note, ha='center', fontsize=8, color='#555555')
    fig.subplots_adjust(left=0.065, right=0.985, top=0.70, bottom=0.135, wspace=0.16)
    return save(fig, out, 'Figure05_NSE_by_horizon', dpi)


CMAP = LinearSegmentedColormap.from_list('nse', ['#8c2b1e', '#f4eeea', '#086b47'])
ROWS6 = ['Seasonal naive', 'Persistence', 'SARIMA', 'XGBoost', 'LSTM', 'Transformer', 'CNN-LSTM', 'MLP']


def fig06(I, out, dpi):
    base_rc()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
    fig.suptitle('Forecast-skill landscape: green = beats historical mean, red = worse', fontsize=12, fontweight='bold', y=0.985)
    for a, tgt, ttl in [(ax[0], 'Fabbisogni', 'A   Demand - NSE by model x horizon'), (ax[1], 'Perdite', 'B   Losses - NSE by model x horizon')]:
        q = I.cfg[(I.cfg.experiment == 'long_term_multivariate') & (I.cfg.target == tgt)]
        M = np.array([[q[(q.model == m) & (q.horizon == h)].NSE_mean.iloc[0] for h in range(1, 13)] for m in ROWS6])
        im = a.imshow(np.clip(M, -1, 1), cmap=CMAP, vmin=-1, vmax=1, aspect='auto')
        for r in range(M.shape[0]):
            for c in range(12):
                v = M[r, c]
                a.text(c, r, f'{v:.2f}' if abs(v) < 1 else f'{v:.0f}', ha='center', va='center', fontsize=6.6,
                       color='white' if (v > 0.56 or v < -0.355) else '#1a1a1a')
        a.set_yticks(range(len(ROWS6))); a.set_yticklabels([m.replace('Seasonal naive', 'Seasonal-naive') for m in ROWS6], fontsize=8)
        a.set_xticks(range(12)); a.set_xticklabels(range(1, 13), fontsize=8.5)
        a.set_xlabel('horizon (months ahead)', fontsize=9.5); a.set_title(ttl, loc='left', fontsize=10.4)
        for sp in a.spines.values():
            sp.set_color('#666666')
    fig.subplots_adjust(left=0.085, right=0.905, top=0.86, bottom=0.12, wspace=0.27)
    cax = fig.add_axes([0.918, 0.12, 0.013, 0.74])
    cb = fig.colorbar(im, cax=cax, extend='min', ticks=np.arange(-1, 1.01, 0.25), format='%.2f')
    cb.set_label('NSE (clipped to -1)', fontsize=9); cb.ax.tick_params(labelsize=10); cb.outline.set_edgecolor('#888888')
    return save(fig, out, 'Figure06_skill_heatmap', dpi)


# ---------------------------------------------------------------- Fig 7 / 8
MOD8 = ['Seasonal naive', 'Persistence', 'SARIMA', 'XGBoost', 'LSTM', 'Transformer', 'CNN-LSTM', 'MLP']
MCOL = {'Seasonal naive': '#2e2e2e', 'Persistence': '#929292', 'SARIMA': '#2e8bc0', 'XGBoost': '#2eaf8c', 'LSTM': '#dc7b2e',
        'Transformer': '#d591b7', 'CNN-LSTM': '#9f7866', 'MLP': '#7979dd'}


def fig_scatter(I, out, dpi, exp, title, name):
    base_rc()
    P = I.pred
    fig, axes = plt.subplots(2, 8, figsize=(15.3, 6.9))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.975)
    for r, (tgt, rl) in enumerate([('Fabbisogni', 'Civil water demand\npred. (×10⁶ m³)'), ('Perdite', 'Network losses\npred. (×10⁶ m³)')]):
        d = P[(P.target == tgt) & (P.experiment == exp) & (P.horizon == 1)]
        mm = I.cfg[(I.cfg.target == tgt) & (I.cfg.experiment == exp) & (I.cfg.horizon == 1)].set_index('model').NSE_mean
        for ci, m in enumerate(MOD8):
            a = axes[r, ci]
            g = d[d.model == m].groupby('origin_idx').agg(obs=('obs', 'first'), pred=('pred', 'mean'))  # 5-seed mean per origin
            o, p = g.obs.values / 1e6, g.pred.values / 1e6
            lo, hi = min(o.min(), p.min()), max(o.max(), p.max()); pad = (hi - lo) * 0.05
            a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '--', color='#bbbbbb', lw=1, zorder=1)
            a.scatter(o, p, s=22, color=MCOL[m], edgecolors='white', linewidths=0.5, zorder=3)
            a.set_xlim(lo - pad, hi + pad); a.set_ylim(lo - pad, hi + pad); a.set_box_aspect(1)
            a.tick_params(labelsize=6.8)
            v = mm[m]
            a.text(0.06, 0.93, f'NSE={v:+.2f}', transform=a.transAxes, fontsize=9.2, fontweight='bold', va='top',
                   color='#127a3e' if v > 0 else '#b03030')
            if r == 0:
                a.set_title(m, fontsize=10.5)
            if ci == 0:
                a.set_ylabel(rl, fontsize=9.5)
            if r == 1:
                a.set_xlabel('obs (×10⁶ m³)', fontsize=8.5)
    fig.text(0.5, 0.02, 'Predicted vs. observed at 1-month lead; markers = 5-seed-mean forecast per origin; dashed = 1:1; '
             'NSE = 5-seed mean. Real data — ATO1 2017–2021.', ha='center', fontsize=8.5, color='#555555')
    fig.subplots_adjust(left=0.045, right=0.99, top=0.86, bottom=0.12, wspace=0.28, hspace=0.22)
    return save(fig, out, name, dpi)


def fig07(I, out, dpi):
    return fig_scatter(I, out, dpi, 'short_term_multivariate',
                       'Observed vs. modelled — short-term forecast (3 → 2 months), multivariate scheme (5 seeds)',
                       'Figure07_obs_vs_modelled_short')


def fig08(I, out, dpi):
    return fig_scatter(I, out, dpi, 'long_term_multivariate',
                       'Observed vs. modelled — long-term forecast (12 → 12 months), multivariate scheme (5 seeds)',
                       'Figure08_obs_vs_modelled_long')


# ---------------------------------------------------------------- Fig 9 / 11
HCOL = {'Seasonal naive': '#1f1f1f', 'XGBoost': '#2ca089', 'LSTM': '#e08214'}


def fig_hindcast(I, out, dpi, tgt, ylab, name, suptitle=None):
    base_rc()
    obs = I.dataset.set_index('Data')[tgt] / 1e6
    d = I.pred[(I.pred.target == tgt) & (I.pred.experiment == 'short_term_multivariate') & (I.pred.horizon == 1)]
    ev0 = pd.Timestamp('2019-01-01')
    fig, axes = plt.subplots(3, 1, figsize=(10.9, 9.2 if suptitle else 8.3), sharex=True)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=0.985)
    x0, x1 = pd.Timestamp('2016-12-18'), pd.Timestamp('2021-12-24')
    for a, m in zip(axes, ['Seasonal naive', 'XGBoost', 'LSTM']):
        c = HCOL[m]
        g = d[d.model == m].groupby('target_date').pred.agg(['mean', 'std']).sort_index()
        ob = d[d.model == m].groupby('target_date').obs.first().sort_index()
        a.axvspan(x0, ev0, color='#e6e6e6', zorder=0, lw=0)
        a.axvline(ev0, color='#888888', ls='--', lw=1.1, zorder=1)
        a.plot(obs.index, obs.values, color='#444444', lw=1.8, zorder=3, label='Observed')
        a.plot(obs.index, obs.values, 'o', color='#444444', ms=4, zorder=3)
        if g['std'].fillna(0).max() > 1e-9:  # band = ±1 SD across seeds (zero for deterministic baselines)
            a.fill_between(g.index, (g['mean'] - g['std']) / 1e6, (g['mean'] + g['std']) / 1e6, color=c, alpha=0.2, lw=0, zorder=2)
        a.plot(g.index, g['mean'] / 1e6, color=c, lw=2.2, zorder=4, label=f'{m} (1-month-ahead hindcast)')
        v = nse(ob.values, g['mean'].values)  # NSE of the seed-mean series
        a.text(0.012, 0.86, f'{m}   NSE = {v:+.2f}', transform=a.transAxes, fontsize=11, fontweight='bold', color=c, va='center',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=c, lw=1.2, alpha=0.95), zorder=6)
        a.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
        a.set_ylabel(ylab, fontsize=9.6); a.grid(alpha=0.3); a.tick_params(labelsize=9.5)
        a.set_xlim(x0, x1)
    trans = mtrans.blended_transform_factory(axes[0].transData, axes[0].transAxes)
    axes[0].text(ev0 + pd.Timedelta(days=22), 0.985, 'evaluation (rolling origin, walk-forward)', transform=trans, fontsize=8.5,
                 color='#595959', va='top', ha='left')
    axes[2].text(0.012, 0.03, 'training / warm-up (before first rolling origin)', transform=axes[2].transAxes, fontsize=8.5,
                 color='#595959', va='bottom', bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.5))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    axes[-1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, p: mdates.num2date(x).strftime('%Y') if mdates.num2date(x).month == 1 else ''))
    axes[-1].set_xlabel('Month', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955 if suptitle else 1], h_pad=0.8)
    return save(fig, out, name, dpi)


def fig09(I, out, dpi):
    return fig_hindcast(I, out, dpi, 'Fabbisogni', 'Civil demand\n(×10⁶ m³)', 'Figure09_hindcast_demand')


def fig11(I, out, dpi):
    return fig_hindcast(I, out, dpi, 'Perdite', 'Network losses\n(×10⁶ m³)', 'Figure11_hindcast_losses',
                        suptitle='Observed vs modelled over time — network losses (the same three methods as Fig. 9)\n'
                                 'one-month-ahead hindcast, short-horizon multivariate, mean of five seeds (band = ±1 SD)')


# ---------------------------------------------------------------- Fig 10
def fig10(I, out, dpi):
    base_rc()
    fig, a = plt.subplots(figsize=(8.3, 3.55))
    _, sn = long_series(I, 'Fabbisogni', 'Seasonal naive')
    a.axhspan(0, 0.3, color=GBAND, zorder=0); a.axhspan(-1.2, 0, color=RBAND, zorder=0)
    a.axhline(0, color='#1a1a1a', lw=1.6, zorder=1)
    for m, mk in [('XGBoost', 'D'), ('LSTM', 'o'), ('SARIMA', '^'), ('Transformer', 'v')]:
        h, v = long_series(I, 'Fabbisogni', m)
        a.plot(h, np.maximum(v - sn, -1.2), color=LINES[m][1], lw=2.2, marker=mk, ms=5.5 if mk != 'D' else 5.0,
               mec='white', mew=0.8, label=m, zorder=3, clip_on=False)
    a.set_ylim(-1.2, 0.35); a.set_xlim(1, 12); a.set_xticks(range(1, 13))
    a.set_title('No trained model reliably beats the free benchmark (demand, parsimonious spec)', loc='left', fontsize=10)
    a.set_ylabel('dNSE (model - seasonal-naive)'); a.set_xlabel('horizon (months ahead)')
    a.legend(loc='lower center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.02))
    a.spines[['top', 'right']].set_visible(False); grey(a)
    fig.tight_layout()
    return save(fig, out, 'Figure10_deltaNSE_vs_seasonalnaive', dpi)


# ---------------------------------------------------------------- Fig 12
def lstm_dem(tbl, h):
    q = tbl[(tbl.model == 'LSTM') & (tbl.target == 'Fabbisogni') & (tbl.experiment == 'short_term_multivariate') & (tbl.horizon == h)].iloc[0]
    return q.NSE_mean, q.NSE_std


def fig12(I, out, dpi):
    base_rc()
    groups = [('7 features\n(minimal)', I.v2), ('9 features\n(parsimonious)', I.cfg), ('14 features\n(extended)', I.p2)]
    fig, a = plt.subplots(figsize=(8.4, 5.6))
    x = np.arange(3); w = 0.36
    for k, (h, lab, c) in enumerate([(1, '1 month ahead', VERM), (2, '2 months ahead', BLUE)]):
        ms = [lstm_dem(t, h) for _, t in groups]
        m = np.array([u for u, _ in ms]); s = np.array([u for _, u in ms])
        xx = x + (k - 0.5) * w
        a.bar(xx, m, w, color=c, label=lab, zorder=2)
        a.errorbar(xx, m, yerr=s, fmt='none', ecolor='#333333', elinewidth=1.6, capsize=6, capthick=1.6, zorder=3)
        for xi, mi, si in zip(xx, m, s):
            a.text(xi, mi + si + 0.045, f'{mi:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    a.axhline(0, color='#333333', lw=1.2, zorder=1)
    a.set_xticks(x); a.set_xticklabels([g for g, _ in groups], fontsize=12)
    a.set_ylabel('LSTM demand NSE', fontsize=12); a.tick_params(axis='y', labelsize=12)
    a.set_ylim(-0.25, 0.95); a.set_title('Parsimony: adding covariates collapses skill', loc='left', fontsize=14)
    a.legend(loc='upper right', frameon=False, fontsize=12)
    a.spines[['top', 'right']].set_visible(False)
    fig.text(0.53, 0.015, 'Same LSTM, same task; only the input feature set changes. Error bars = ±1 SD across 5 seeds. Real data.',
             ha='center', fontsize=10, color='#555555')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return save(fig, out, 'Figure12_feature_parsimony', dpi)


# ---------------------------------------------------------------- Fig 13
def fig13(I, out, dpi):
    base_rc()
    q = I.cfg[(I.cfg.target == 'Fabbisogni') & (I.cfg.experiment == 'short_term_multivariate') & (I.cfg.horizon == 1)].copy()
    q = q.sort_values('NSE_mean', ascending=False).reset_index(drop=True)
    lab = {'Seasonal naive': 'Seasonal-naïve'}
    fig, a = plt.subplots(figsize=(10.0, 5.3))
    n = len(q); ys = np.arange(n)[::-1]
    lo = (q.NSE_mean - q.NSE_std).min(); tx = (q.NSE_mean + q.NSE_std).max() + 0.7
    for y, (_, r) in zip(ys, q.iterrows()):
        m, s = r.NSE_mean, r.NSE_std  # NSE_std = sample SD across the five seeds
        col = '#999999' if r.model in ('Seasonal naive', 'Persistence') else (GREEN if m > 0 else RED)
        if s > 1e-9:
            a.hlines(y, m - s, m + s, color='#aaaaaa', lw=3.2, zorder=2, capstyle='round')
        a.scatter([m], [y], s=150, color=col, edgecolors='white', linewidths=1.3, zorder=3)
        a.text(tx, y, f'{m:.2f} ± {s:.2f}', va='center', ha='left', fontsize=11, family='monospace')
    a.axvline(0, color='#1a1a1a', lw=1.5, zorder=1)
    a.set_yticks(ys); a.set_yticklabels([lab.get(mm, mm) for mm in q.model], fontsize=11.8)
    a.tick_params(axis='y', length=0); a.tick_params(axis='x', labelsize=12)
    a.set_xlim(lo - 0.5, tx + 3.0); a.set_ylim(-0.35, n - 0.6)
    a.set_xticks([t for t in range(int(np.floor(lo)) - 1, 1) if t % 2 == 0 and t >= lo - 0.5])
    a.set_xlabel('NSE (mean ± SD across 5 seeds)', fontsize=12)
    a.set_title('Reliability: seed dispersion by model (demand, 1-month)', loc='left', fontsize=13.8)
    a.spines[['top', 'right', 'left']].set_visible(False)
    fig.tight_layout()
    return save(fig, out, 'Figure13_reliability_seed_dispersion', dpi)


FIGS = {2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06, 7: fig07, 8: fig08, 9: fig09, 10: fig10, 11: fig11, 12: fig12, 13: fig13}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--only', nargs='*', type=int, help='figure numbers to build (default: all)')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--titles', action='store_true', help='also draw figure-level titles and footnotes')
    ap.add_argument('--out', type=Path, default=ROOT / 'figures' / 'output')
    ap.add_argument('--data', type=Path, default=ROOT / 'data' / 'Dataset.xlsx', help='operator dataset (withheld)')
    ap.add_argument('--predictions', type=Path, default=ROOT / 'output_configc' / 'configc_predictions_long.csv',
                    help='parsimonious-pipeline prediction table (withheld)')
    args = ap.parse_args(argv)
    global SHOW_TITLES
    SHOW_TITLES = args.titles
    I = Inputs(args.data, args.predictions)
    made, skipped = [], []
    for k in (args.only or sorted(FIGS)):
        try:
            p = FIGS[k](I, args.out, args.dpi)
            made.append(k); print(f'Fig. {k:2d}: {p}')
        except FileNotFoundError as e:
            skipped.append(k); print(f'Fig. {k:2d}: skipped — input not available ({e}); see data/README_DATA.md')
    print(f'\n{len(made)} figure(s) written to {args.out}; {len(skipped)} skipped.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
