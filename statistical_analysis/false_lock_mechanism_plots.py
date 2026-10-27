"""dToF false-lock mechanism figures from the statistical_analysis/metrics CSVs written by run_analysis.py."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plot_style import (
    setup_ieee_style, save_fig, IEEE_SINGLE_COL, IEEE_DOUBLE_COL, TOL_MUTED,
    DISTANCE_COLORS, TRIAL_COLORS, CONC_AXIS_LABEL_SHORT,
)

METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

MECHANISM_WINDOW = 'initial_30s'
MECHANISM_LOADING = 0.8

STATE_COLORS = {'false_lock': TOL_MUTED['rose'],
                'intermediate': TOL_MUTED['sand'],
                'wall_like': TOL_MUTED['teal']}
STATE_LABELS = {'false_lock': 'False lock',
                'intermediate': 'Intermediate',
                'wall_like': 'Wall-like'}


def _read(name):
    return pd.read_csv(os.path.join(METRICS_DIR, name))


def _save(fig, stem):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_fig(fig, os.path.join(FIGURES_DIR, f'{stem}.png'))
    plt.close(fig)


def plot_normalized_distributions(distances=(30, 40, 50, 60)):
    """Fig S1: time-weighted q = D_rep/D distributions, 0-30 s, 0.8 g (trials + mean)."""
    setup_ieee_style()
    by_run = _read('false_lock_histograms_by_run.csv')
    cond = _read('false_lock_histograms_condition_summary.csv')
    sel = lambda df: ((df['window'] == MECHANISM_WINDOW)
                      & (df['loading_g'] == MECHANISM_LOADING))
    by_run, cond = by_run[sel(by_run)], cond[sel(cond)]

    fig, axes = plt.subplots(2, 2, figsize=(IEEE_DOUBLE_COL, 3.8), sharex=True)
    for i, (ax, dist) in enumerate(zip(axes.ravel(), distances)):
        runs = by_run[by_run['distance_cm'] == dist]
        cmean = cond[cond['distance_cm'] == dist].sort_values('q_bin_low')
        lows = cmean['q_bin_low'].values
        edges = np.append(lows, lows[-1] + (lows[1] - lows[0]))

        y_max = float(cmean['probability'].max())
        for idx, trial in enumerate((1, 2, 3)):
            r = runs[runs['trial'] == trial].sort_values('q_bin_low')
            y_max = max(y_max, float(r['probability'].max()))
            ax.stairs(r['probability'].values, edges,
                      color=TRIAL_COLORS[idx], lw=0.6, alpha=0.6, zorder=2)
        ax.stairs(cmean['probability'].values, edges, color='black', lw=1.0, zorder=3)
        for q, ls in ((0.70, '--'), (0.90, '--'), (1.00, ':')):
            ax.axvline(q, color='0.4', ls=ls, lw=0.7, zorder=1)

        ax.set_xlim(0.0, 1.2)
        ax.set_ylim(0.0, max(0.05, 1.08 * y_max))
        ax.set_title(f'{dist} cm')
        if i >= 2:
            ax.set_xlabel(r'$q = D_{\mathrm{rep}}/D$')
        if i % 2 == 0:
            ax.set_ylabel('Probability per 0.01 q')

    handles = [
        Line2D([], [], color='0.5', lw=0.6, label='Trial'),
        Line2D([], [], color='black', lw=1.0, label='Mean'),
        Line2D([], [], color='0.4', ls='--', lw=0.7, label='q = 0.70, 0.90'),
        Line2D([], [], color='0.4', ls=':', lw=0.7, label='q = 1.00'),
    ]
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    _save(fig, 'fig_s1_normalized_distance_distributions')


def plot_state_occupancy():
    """Fig S2: false-lock / intermediate / wall-like occupancy per C_air bin, 0.8 g."""
    setup_ieee_style()
    df = _read('false_lock_state_occupancy_vs_concentration.csv')
    df = df[(df['level'] == 'condition_mean') & (df['loading_g'] == MECHANISM_LOADING)
            & (df['distance_cm'].isin([40, 50, 60]))]
    fig, axes = plt.subplots(1, 3, figsize=(IEEE_DOUBLE_COL, 2.2), sharey=True)
    for ax, dist in zip(axes, (40, 50, 60)):
        sub = df[df['distance_cm'] == dist].sort_values('conc_bin_low_g_m3')
        lo, hi = sub['conc_bin_low_g_m3'].values, sub['conc_bin_high_g_m3'].values
        centers = (lo + hi) / 2.0
        bottom = np.zeros(len(sub))
        for state in ('false_lock', 'intermediate', 'wall_like'):
            vals = sub[f'{state}_fraction'].values
            ax.bar(centers, vals, width=(hi - lo) * 0.9, bottom=bottom,
                   color=STATE_COLORS[state], edgecolor='white', linewidth=0.3)
            bottom += vals
        ax.plot(centers, sub['invalid_fraction'].values, color='black', lw=0.7, ls=':',
                marker='.', ms=2)
        ax.set_title(f'{dist} cm')
        ax.set_xlabel(CONC_AXIS_LABEL_SHORT)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel('State occupancy')
    handles = [Patch(color=STATE_COLORS[s], label=STATE_LABELS[s]) for s in STATE_COLORS]
    handles.append(Line2D([], [], color='black', lw=0.7, ls=':', marker='.', ms=2,
                          label='Invalid'))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    _save(fig, 'fig_s2_state_occupancy')


def plot_threshold_sensitivity():
    """Fig S3: initial 30 s false-lock fraction at 0.60/0.70/0.80 D thresholds, 0.8 g."""
    setup_ieee_style()
    df = _read('false_lock_threshold_sensitivity.csv')
    df = df[(df['window'] == MECHANISM_WINDOW) & (df['loading_g'] == MECHANISM_LOADING)
            & (df['distance_cm'].isin([30, 40, 50, 60]))]
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.5))
    for dist in (30, 40, 50, 60):
        color = DISTANCE_COLORS[dist]
        runs = df[(df['level'] == 'run') & (df['distance_cm'] == dist)]
        for trial in (1, 2, 3):
            r = runs[runs['trial'] == trial]
            ax.plot(r['threshold_frac'] + (trial - 2) * 0.006, r['false_lock_fraction'],
                    ls='', marker='o', ms=2.0, color=color, alpha=0.5, zorder=2)
        cond = df[(df['level'] == 'condition_mean')
                  & (df['distance_cm'] == dist)].sort_values('threshold_frac')
        ax.plot(cond['threshold_frac'], cond['false_lock_fraction'], color=color,
                lw=1.0, marker='s', ms=3.0, zorder=3, label=f'{dist} cm')
    ax.set_xticks([0.60, 0.70, 0.80])
    ax.set_xlim(0.56, 0.84)
    ax.set_xlabel(r'False lock threshold (fraction of $D$)')
    ax.set_ylabel('False lock fraction (0-30 s)')
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=6,
               bbox_to_anchor=(0.5, -0.01))
    _save(fig, 'fig_s3_threshold_sensitivity')


def plot_false_return_range():
    """Fig S4: per-trial quantiles of the reported distance of false-lock readings, 0.8 g."""
    setup_ieee_style()
    df = _read('false_return_range_run_summary.csv')
    df = df[(df['window'] == MECHANISM_WINDOW) & (df['loading_g'] == MECHANISM_LOADING)
            & (df['distance_cm'].isin([40, 50, 60]))]
    df = df[np.isfinite(df['abs_range_cm_p50'])]
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.5))
    for _, r in df.iterrows():
        x = r['distance_cm'] + (r['trial'] - 2) * 1.6
        color = DISTANCE_COLORS[int(r['distance_cm'])]
        ax.plot([x, x], [r['abs_range_cm_p10'], r['abs_range_cm_p90']],
                color=color, lw=0.7, alpha=0.8)
        ax.plot([x, x], [r['abs_range_cm_p25'], r['abs_range_cm_p75']],
                color=color, lw=2.0, alpha=0.9)
        ax.plot(x, r['abs_range_cm_p50'], marker='o', ms=3.5, color=color,
                mec='black', mew=0.4, zorder=3)
    ax.set_xticks([40, 50, 60])
    ax.set_xlabel('True distance (cm)')
    ax.set_ylabel('False lock reported distance (cm)')
    ax.set_ylim(bottom=0)
    handles = [
        Line2D([], [], color='0.3', marker='o', ms=3.5, ls='', label='Trial median'),
        Line2D([], [], color='0.3', lw=2.0, label='p25-p75'),
        Line2D([], [], color='0.3', lw=0.7, label='p10-p90'),
    ]
    ax.legend(handles=handles, fontsize=6, loc='upper right')
    fig.tight_layout()
    _save(fig, 'fig_s4_false_return_range')


def generate_plots():
    plot_normalized_distributions()
    plot_state_occupancy()
    plot_threshold_sensitivity()
    plot_false_return_range()


if __name__ == '__main__':
    generate_plots()
