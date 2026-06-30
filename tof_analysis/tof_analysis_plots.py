"""Generate ToF model vs experimental comparison figures from tof_analysis.run_analysis() results."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import (
    setup_ieee_style, save_fig,
    IEEE_SINGLE_COL, IEEE_DOUBLE_COL,
    TOL_MUTED, DISTANCE_COLORS, TRIAL_COLORS, LOADING_COLORS,
    PANEL_ORDER, PANEL_LABELS, PANEL_YLIMS,
)
from matplotlib.patches import Patch

from tof_analysis import (
    model_stochastic, load_raw_center_readings,
    DISTANCES_CM, LOADINGS_G, TRIALS,
    FL_THRESHOLD_FRAC,
)

DIST_COLORS = DISTANCE_COLORS
YLIMS = PANEL_YLIMS

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')


def plot_scatter_overlay(all_data, scalar_metrics):
    """Fig 1: 3x2 scatter overlay of exp + model stochastic at 0.8g."""
    setup_ieee_style()

    grams = 0.8
    t_dense = np.linspace(0, 10, 200)
    t_sparse = np.linspace(10, 15, 30)
    T_MIN = np.unique(np.concatenate([t_dense, t_sparse]))
    t_grid_s = T_MIN * 60.0
    n_frames = 80

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    axes = axes.flatten()

    for idx, (dist_cm, label) in enumerate(zip(PANEL_ORDER, PANEL_LABELS)):
        ax = axes[idx]

        for trial_idx, trial in enumerate(TRIALS):
            t_s, vals = load_raw_center_readings(grams, dist_cm, trial)
            if t_s is None:
                continue
            t_min = t_s / 60.0
            for col_data in vals.values():
                ax.scatter(t_min, col_data, s=0.5, alpha=0.35,
                           color=TRIAL_COLORS[trial_idx],
                           linewidths=0, zorder=2)

        print(f"    Running stochastic sim {grams}g @ {dist_cm}cm "
              f"({len(t_grid_s)} pts x {n_frames} frames)...")
        sim_cm = model_stochastic(dist_cm, grams, t_grid_s, n_frames=n_frames)
        t_sim_min = np.repeat(T_MIN, n_frames)
        sim_flat = sim_cm.flatten()
        valid = ~np.isnan(sim_flat)
        ax.scatter(t_sim_min[valid], sim_flat[valid],
                   color=TOL_MUTED['rose'], s=0.5, alpha=0.35,
                   linewidths=0, zorder=3)

        ax.axhline(dist_cm, color='#BBCC33', ls='-', lw=0.8, zorder=4)

        row = scalar_metrics[
            (scalar_metrics['loading_g'] == grams) &
            (scalar_metrics['dist_cm'] == dist_cm)
        ]
        if len(row) > 0 and not row.iloc[0].get('is_always_wall_locked_mod', False):
            snap_min = row.iloc[0].get('snap_time_mod_min', np.nan)
            if not np.isnan(snap_min):
                ax.axvline(snap_min, color='#e06000', ls='--', lw=0.8,
                           alpha=0.7, zorder=5)

        y_lo, y_hi = YLIMS[dist_cm]
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(-0.3, 15.3)
        ax.set_xlabel('Time (min)', fontsize=9)
        ax.set_ylabel('Distance (cm)', fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D([], [], color=TRIAL_COLORS[0], ls='', marker='o', markersize=3,
               label='Trial 1'),
        Line2D([], [], color=TRIAL_COLORS[1], ls='', marker='o', markersize=3,
               label='Trial 2'),
        Line2D([], [], color=TRIAL_COLORS[2], ls='', marker='o', markersize=3,
               label='Trial 3'),
        Line2D([], [], color=TOL_MUTED['rose'], ls='', marker='o',
               markersize=3, label='Model (stochastic)'),
        Line2D([], [], color='#BBCC33', ls='-', lw=0.8,
               label='True distance'),
        Line2D([], [], color='#e06000', ls='--', lw=0.8,
               label='Model snap time'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=6,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))

    from tof_model import ALPHA_0_MEASURED, S_LIDAR as _S
    fig.suptitle(
        f'Model vs Experiment — {grams} g Arizona ISO A0\n'
        f'ALPHA_0 = {ALPHA_0_MEASURED}, S_LIDAR = {_S} sr',
        fontsize=10, y=1.01)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_a1_scatter.png')
    save_fig(fig, path)
    plt.close(fig)
    print(f"  Saved {path}")


def _dense_model_curves(dist_cm, mass_g, n_frames=80):
    """Smooth model f_FL and bias% curves on a 2s-step grid; returns dict of arrays."""
    from tof_analysis import run_model_stochastic
    t_grid_s = np.arange(0, 900 + 2, 2.0)
    result = run_model_stochastic(dist_cm, mass_g, t_grid_s, n_frames=n_frames)

    t_min = t_grid_s / 60.0
    n_times = len(t_grid_s)
    f_fl = np.full(n_times, np.nan)
    bias_pct = np.full(n_times, np.nan)
    bias_pct_all = []
    threshold_cm = dist_cm * FL_THRESHOLD_FRAC

    for i in range(n_times):
        row = result['distances_cm'][i, :]
        valid = ~np.isnan(row)
        if not valid.any():
            continue
        vals = row[valid]
        fl = vals < threshold_cm
        f_fl[i] = fl.mean()
        wall_vals = vals[~fl]
        if len(wall_vals) > 0:
            bias_pct[i] = (np.median(wall_vals) - dist_cm) / dist_cm * 100.0
            for v in wall_vals:
                bias_pct_all.append((t_min[i], (v - dist_cm) / dist_cm * 100.0))

    if len(bias_pct_all) > 0:
        bp_arr = np.array(bias_pct_all)
        order = np.argsort(bp_arr[:, 0])
        bp_sorted = bp_arr[order]
        kern = min(512, len(bp_sorted))
        bp_vals = pd.Series(bp_sorted[:, 1])
        bp_smooth = bp_vals.rolling(kern, min_periods=1, center=True).mean().values
        bp_std = bp_vals.rolling(kern, min_periods=1, center=True).std().values
        bias_pct_smooth = np.interp(t_min, bp_sorted[:, 0], bp_smooth)
        bias_pct_std = np.interp(t_min, bp_sorted[:, 0], bp_std)
    else:
        bias_pct_smooth = np.full(n_times, np.nan)
        bias_pct_std = np.full(n_times, np.nan)

    return {
        't_min': t_min, 'f_fl': f_fl, 'bias_pct': bias_pct,
        'bias_pct_smooth': bias_pct_smooth, 'bias_pct_std': bias_pct_std,
    }


_dense_cache = {}


def _get_dense(dist_cm, mass_g):
    key = (dist_cm, mass_g)
    if key not in _dense_cache:
        _dense_cache[key] = _dense_model_curves(dist_cm, mass_g)
    return _dense_cache[key]


def plot_false_lock_comparison(ts_metrics, all_data=None):
    """Fig 2: f_FL(t) curves, exp vs model, 0.8g only."""
    setup_ieee_style()

    grams = 0.8
    kern = 512
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.5))
    plotted_dists = []

    for dist_cm in DISTANCES_CM:
        sub = ts_metrics[(ts_metrics['loading_g'] == grams) &
                         (ts_metrics['dist_cm'] == dist_cm)]
        if len(sub) == 0:
            continue

        dense = _get_dense(dist_cm, grams)
        max_fl_exp = sub['f_fl_exp'].max()
        max_fl_mod = sub['f_fl_mod'].max() if 'f_fl_mod' in sub.columns else 0
        if dist_cm < 35 or max(max_fl_exp, max_fl_mod) < 0.005:
            continue

        plotted_dists.append(dist_cm)
        color = DIST_COLORS[dist_cm]

        if all_data is not None:
            trial_curves = []
            for trial in TRIALS:
                key = (grams, trial, dist_cm)
                df = all_data.get(key)
                if df is None or len(df) == 0:
                    continue
                df_sorted = df.sort_values('time_seconds')
                t_min_arr = df_sorted['time_seconds'].values / 60.0
                fl_raw = df_sorted['is_false_lock'].astype(float).values
                if len(fl_raw) < kern:
                    continue
                fl_smooth = pd.Series(fl_raw).rolling(
                    kern, min_periods=1, center=True).mean().values
                trial_curves.append((t_min_arr, fl_smooth))

            if len(trial_curves) > 1:
                t_lo = max(tc.min() for tc, _ in trial_curves)
                t_hi = min(tc.max() for tc, _ in trial_curves)
                common_t = np.linspace(t_lo, t_hi, 300)
                stacked = np.column_stack([
                    np.interp(common_t, tc, fl)
                    for tc, fl in trial_curves
                ])
                mean_fl = stacked.mean(axis=1)
                ax.plot(common_t, mean_fl, color=color, ls='-', lw=1.0)
            elif len(trial_curves) == 1:
                tc, fl = trial_curves[0]
                ax.plot(tc, fl, color=color, ls='-', lw=1.0)
        else:
            ax.plot(sub['time_bin_center_min'], sub['f_fl_exp'],
                    color=color, ls='-', lw=1.0)

        ax.plot(dense['t_min'], dense['f_fl'],
                color=color, ls='--', lw=1.0)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, 5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('False lock fraction')

    legend_handles = []
    for dist_cm in plotted_dists:
        color = DIST_COLORS[dist_cm]
        legend_handles.append(
            Line2D([], [], color=color, ls='-', lw=1.0,
                   label=f'{dist_cm} cm experiment'))
        legend_handles.append(
            Line2D([], [], color=color, ls='--', lw=1.0,
                   label=f'{dist_cm} cm model'))
    ax.legend(handles=legend_handles, fontsize=6, loc='upper right')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_a2_false_lock.png')
    save_fig(fig, path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_bias_recovery(ts_metrics, all_data=None):
    """Fig 3: 2x6 grid — wall-locked % error vs time (rows=loadings, cols=distances)."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    setup_ieee_style()

    n_cols = len(DISTANCES_CM)
    fig, axes = plt.subplots(2, n_cols, figsize=(IEEE_DOUBLE_COL, 3.2),
                             sharex=True)

    YLIMS_PCT = {
        0.2: {10: (-2, 2), 20: (-2, 5), 30: (-4, 4),
              40: (-2, 3), 50: (-1, 1), 60: (-1, 1)},
        0.8: {10: (-4, 8), 20: (-17, 1), 30: (-10, 1),
              40: (-3, 1), 50: (-1, 1), 60: (-1, 1)},
    }
    YFMT = {10: '%.0f', 20: '%.0f', 30: '%.0f',
            40: '%.0f', 50: '%.1f', 60: '%.1f'}

    kern = 512
    MODEL_COLOR = '#444444'

    for row_idx, grams in enumerate(LOADINGS_G):
        color = LOADING_COLORS[grams]
        ylims_g = YLIMS_PCT.get(grams, YLIMS_PCT[0.8])

        for col_idx, dist_cm in enumerate(DISTANCES_CM):
            ax = axes[row_idx, col_idx]

            if all_data is not None:
                trial_curves = []
                for trial in TRIALS:
                    key = (grams, trial, dist_cm)
                    df = all_data.get(key)
                    if df is None or len(df) == 0:
                        continue
                    wall = df[~df['is_false_lock']].copy()
                    if len(wall) < kern:
                        continue
                    wall = wall.sort_values('time_seconds')
                    t_min_arr = wall['time_seconds'].values / 60.0
                    err_pct_raw = (wall['distance_cm'].values - dist_cm) / dist_cm * 100.0
                    err_smooth = pd.Series(err_pct_raw).rolling(
                        kern, min_periods=1, center=True).mean().values
                    trial_curves.append((t_min_arr, err_smooth))

                if len(trial_curves) > 1:
                    t_lo = max(tc.min() for tc, _ in trial_curves)
                    t_hi = min(tc.max() for tc, _ in trial_curves)
                    common_t = np.linspace(t_lo, t_hi, 300)
                    stacked = np.column_stack([
                        np.interp(common_t, tc, err)
                        for tc, err in trial_curves
                    ])
                    mean_err = stacked.mean(axis=1)
                    std_err = stacked.std(axis=1)
                    ax.plot(common_t, mean_err, color=color, linewidth=0.8)
                    ax.fill_between(common_t, mean_err - std_err,
                                    mean_err + std_err, color=color, alpha=0.15)
                elif len(trial_curves) == 1:
                    tc, err = trial_curves[0]
                    ax.plot(tc, err, color=color, linewidth=0.8)

            dense = _get_dense(dist_cm, grams)
            valid_d = ~np.isnan(dense['bias_pct_smooth'])
            if valid_d.any():
                t_m = dense['t_min'][valid_d]
                m_smooth = dense['bias_pct_smooth'][valid_d]
                m_std = dense['bias_pct_std'][valid_d]
                step = 7
                ax.plot(t_m[::step], m_smooth[::step], color=MODEL_COLOR,
                        linewidth=0.8, ls='--', zorder=10)
                ax.fill_between(t_m, m_smooth - m_std, m_smooth + m_std,
                                color=MODEL_COLOR, alpha=0.12, zorder=9)

            ax.axhline(0, color='0.4', ls='--', lw=0.5)
            if dist_cm in ylims_g:
                ax.set_ylim(ylims_g[dist_cm])
            ax.yaxis.set_major_locator(LinearLocator(numticks=5))
            if dist_cm in YFMT:
                ax.yaxis.set_major_formatter(FormatStrFormatter(YFMT[dist_cm]))

            if row_idx == 0:
                ax.set_title(f'{dist_cm} cm', fontsize=8)
            if row_idx == 1:
                ax.set_xlabel('Time (min)')
            if col_idx == 0:
                ax.set_ylabel(f'{grams} g\nError (%)')

    style_handles = [
        Line2D([], [], color=LOADING_COLORS[0.2], ls='-', lw=0.8,
               label='Exp mean (0.2 g)'),
        Patch(facecolor=LOADING_COLORS[0.2], alpha=0.15,
              label='Exp $\\pm$1 SD (0.2 g)'),
        Line2D([], [], color=LOADING_COLORS[0.8], ls='-', lw=0.8,
               label='Exp mean (0.8 g)'),
        Patch(facecolor=LOADING_COLORS[0.8], alpha=0.15,
              label='Exp $\\pm$1 SD (0.8 g)'),
        Line2D([], [], color=MODEL_COLOR, ls='--', lw=0.8,
               label='Model mean'),
        Patch(facecolor=MODEL_COLOR, alpha=0.12, label='Model $\\pm$1 SD'),
    ]
    fig.legend(handles=style_handles,
               loc='lower center', ncol=6, fontsize=6,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(OUTPUT_DIR, 'fig_a3_bias_recovery.png')
    save_fig(fig, path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_bias_recovery_overlay_aligned(ts_metrics, all_data=None):
    """Fig 3d: 1x6 overlay with zero aligned at 1/4 from top across all panels."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import FormatStrFormatter
    setup_ieee_style()

    n_cols = len(DISTANCES_CM)
    fig, axes = plt.subplots(1, n_cols, figsize=(IEEE_DOUBLE_COL, 1.8))

    kern = 512
    PAD = 1.05

    def _lighten(hex_color, amount=0.4):
        h = hex_color.lstrip('#')
        r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = int(r + (255 - r) * amount)
        g = int(g + (255 - g) * amount)
        b = int(b + (255 - b) * amount)
        return f'#{r:02x}{g:02x}{b:02x}'

    EXP_COLORS = {g: _lighten(LOADING_COLORS[g]) for g in LOADINGS_G}
    MODEL_COLORS = LOADING_COLORS

    panel_yvals = {d: [] for d in DISTANCES_CM}

    for col_idx, dist_cm in enumerate(DISTANCES_CM):
        ax = axes[col_idx]

        for grams in LOADINGS_G:
            exp_color = EXP_COLORS[grams]
            mod_color = MODEL_COLORS[grams]

            if all_data is not None:
                trial_curves = []
                for trial in TRIALS:
                    key = (grams, trial, dist_cm)
                    df = all_data.get(key)
                    if df is None or len(df) == 0:
                        continue
                    wall = df[~df['is_false_lock']].copy()
                    if len(wall) < kern:
                        continue
                    wall = wall.sort_values('time_seconds')
                    t_min_arr = wall['time_seconds'].values / 60.0
                    err_pct_raw = (wall['distance_cm'].values - dist_cm) / dist_cm * 100.0
                    err_smooth = pd.Series(err_pct_raw).rolling(
                        kern, min_periods=1, center=True).mean().values
                    trial_curves.append((t_min_arr, err_smooth))

                if len(trial_curves) > 1:
                    t_lo = max(tc.min() for tc, _ in trial_curves)
                    t_hi = min(tc.max() for tc, _ in trial_curves)
                    common_t = np.linspace(t_lo, t_hi, 300)
                    stacked = np.column_stack([
                        np.interp(common_t, tc, err)
                        for tc, err in trial_curves
                    ])
                    mean_err = stacked.mean(axis=1)
                    std_err = stacked.std(axis=1)
                    ax.plot(common_t, mean_err, color=exp_color, linewidth=0.9)
                    ax.fill_between(common_t, mean_err - std_err,
                                    mean_err + std_err, color=exp_color, alpha=0.18)
                    panel_yvals[dist_cm].extend(mean_err + std_err)
                    panel_yvals[dist_cm].extend(mean_err - std_err)
                elif len(trial_curves) == 1:
                    tc, err = trial_curves[0]
                    ax.plot(tc, err, color=exp_color, linewidth=0.9)
                    panel_yvals[dist_cm].extend(err)

            dense = _get_dense(dist_cm, grams)
            valid_d = ~np.isnan(dense['bias_pct_smooth'])
            if valid_d.any():
                t_m = dense['t_min'][valid_d]
                m_smooth = dense['bias_pct_smooth'][valid_d]
                step = 7
                ax.plot(t_m[::step], m_smooth[::step], color=mod_color,
                        linewidth=0.9, ls='--', zorder=10)
                panel_yvals[dist_cm].extend(m_smooth)

        ax.axhline(0, color='0.4', ls='--', lw=0.5)
        ax.set_title(f'{dist_cm} cm', fontsize=8)
        ax.set_xlabel('Time (min)')
        if col_idx == 0:
            ax.set_ylabel('Error (%)')

    Y_TOP_OVERRIDE = {20: 5, 30: 3, 40: 1}

    for col_idx, dist_cm in enumerate(DISTANCES_CM):
        ax = axes[col_idx]
        vals = np.array(panel_yvals[dist_cm])
        if len(vals) == 0:
            continue
        if dist_cm in Y_TOP_OVERRIDE:
            y_top = Y_TOP_OVERRIDE[dist_cm]
        else:
            data_max = np.nanmax(vals) * PAD
            data_min = np.nanmin(vals) * PAD
            y_top_from_max = max(data_max, 0.01)
            y_top_from_min = abs(data_min) / 3.0 if data_min < 0 else 0.01
            y_top = max(y_top_from_max, y_top_from_min)
        y_bot = -3.0 * y_top
        ax.set_ylim(y_bot, y_top)
        ax.set_xlim(0, 15)
        ax.xaxis.set_major_locator(plt.FixedLocator([0, 5, 10, 15]))
        ax.yaxis.set_major_locator(plt.LinearLocator(5))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    style_handles = [
        Line2D([], [], color=EXP_COLORS[0.2], ls='-', lw=0.9,
               label='0.2 g mean'),
        Patch(facecolor=EXP_COLORS[0.2], alpha=0.18,
              label='0.2 g $\\pm$1 SD'),
        Line2D([], [], color=MODEL_COLORS[0.2], ls='--', lw=0.9,
               label='0.2 g model'),
        Line2D([], [], color=EXP_COLORS[0.8], ls='-', lw=0.9,
               label='0.8 g mean'),
        Patch(facecolor=EXP_COLORS[0.8], alpha=0.18,
              label='0.8 g $\\pm$1 SD'),
        Line2D([], [], color=MODEL_COLORS[0.8], ls='--', lw=0.9,
               label='0.8 g model'),
    ]
    fig.legend(handles=style_handles,
               loc='lower center', ncol=6, fontsize=6,
               bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUTPUT_DIR, 'fig_a4_bias_recovery_aligned.png')
    save_fig(fig, path)
    plt.close(fig)
    print(f"  Saved {path}")


LOADING_COLORS = {0.2: TOL_MUTED['indigo'], 0.8: TOL_MUTED['rose']}


def plot_bias_recovery_wide(ts_metrics, all_data):
    """Fig 3v2: 1x6 wide strip per loading — per-trial % error, model as dark grey band."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    setup_ieee_style()

    YLIMS_PCT = {
        0.8: {10: (-4, 8), 20: (-17, 1), 30: (-10, 1),
              40: (-3, 1), 50: (-1, 1), 60: (-1, 1)},
        0.2: {10: (-2, 2), 20: (-2, 5), 30: (-4, 4),
              40: (-2, 3), 50: (-1, 1), 60: (-1, 1)},
    }
    YFMT = {10: '%.0f', 20: '%.0f', 30: '%.0f',
            40: '%.0f', 50: '%.1f', 60: '%.1f'}
    MODEL_COLOR = '#444444'

    for grams in LOADINGS_G:
        n_cols = len(DISTANCES_CM)
        fig, axes = plt.subplots(1, n_cols, figsize=(IEEE_DOUBLE_COL, 1.8))
        color = LOADING_COLORS[grams]

        for idx, d_cm in enumerate(DISTANCES_CM):
            ax = axes[idx]

            trial_curves = []
            kern = 512
            for trial in TRIALS:
                key = (grams, trial, d_cm)
                df = all_data.get(key)
                if df is None or len(df) == 0:
                    continue
                wall = df[~df['is_false_lock']].copy()
                if len(wall) < kern:
                    continue
                wall = wall.sort_values('time_seconds')
                t_min = wall['time_seconds'].values / 60.0
                err_pct_raw = (wall['distance_cm'].values - d_cm) / d_cm * 100.0
                err_smooth = pd.Series(err_pct_raw).rolling(
                    kern, min_periods=1, center=True).mean().values
                trial_curves.append((t_min, err_smooth))

            for t_c, err in trial_curves:
                ax.plot(t_c, err, color=color, linewidth=0.4, alpha=0.25)

            if len(trial_curves) > 1:
                t_lo = max(tc.min() for tc, _ in trial_curves)
                t_hi = min(tc.max() for tc, _ in trial_curves)
                common_t = np.linspace(t_lo, t_hi, 300)
                stacked = np.column_stack([
                    np.interp(common_t, tc, err)
                    for tc, err in trial_curves
                ])
                mean_err = stacked.mean(axis=1)
                std_err = stacked.std(axis=1)
                ax.plot(common_t, mean_err, color=color, linewidth=0.8)
                ax.fill_between(common_t, mean_err - std_err,
                                mean_err + std_err, color=color, alpha=0.15)
            elif len(trial_curves) == 1:
                tc, err = trial_curves[0]
                ax.plot(tc, err, color=color, linewidth=0.8)

            dense = _get_dense(d_cm, grams)
            valid_d = ~np.isnan(dense['bias_pct_smooth'])
            if valid_d.any():
                t_m = dense['t_min'][valid_d]
                m_smooth = dense['bias_pct_smooth'][valid_d]
                m_std = dense['bias_pct_std'][valid_d]
                ax.plot(t_m, m_smooth, color=MODEL_COLOR, linewidth=0.8,
                        ls='--', zorder=10)
                ax.fill_between(t_m, m_smooth - m_std, m_smooth + m_std,
                                color=MODEL_COLOR, alpha=0.12, zorder=9)

            ax.axhline(0, color='0.4', ls='--', lw=0.5)
            ylims_g = YLIMS_PCT.get(grams, YLIMS_PCT[0.8])
            if d_cm in ylims_g:
                ax.set_ylim(ylims_g[d_cm])
            ax.yaxis.set_major_locator(LinearLocator(numticks=5))
            if d_cm in YFMT:
                ax.yaxis.set_major_formatter(FormatStrFormatter(YFMT[d_cm]))

            ax.set_title(f'{d_cm} cm', fontsize=8)
            ax.set_xlabel('Time (min)')
            if idx == 0:
                ax.set_ylabel('Error (%)')

        style_handles = [
            Line2D([], [], color='0.3', ls='-', lw=0.4, alpha=0.25,
                   label='Exp trial'),
            Line2D([], [], color='0.3', ls='-', lw=0.8, label='Exp mean'),
            Patch(facecolor='0.7', alpha=0.15, label='Exp $\\pm$1 SD'),
            Line2D([], [], color=MODEL_COLOR, ls='-', lw=0.8,
                   label='Model mean'),
            Patch(facecolor=MODEL_COLOR, alpha=0.12, label='Model $\\pm$1 SD'),
        ]
        fig.legend(handles=style_handles,
                   loc='lower center', ncol=5, fontsize=6,
                   bbox_to_anchor=(0.5, -0.02))

        fig.tight_layout(rect=[0, 0.04, 1, 1])
        path = os.path.join(OUTPUT_DIR,
                            f'fig_a5_bias_recovery_wide_{grams}g.png')
        save_fig(fig, path)
        plt.close(fig)
        print(f"  Saved {path}")


def plot_bias_recovery_raw(all_data):
    """Fig 3v3: 1x6 wide strip per loading — raw wall-locked bias scatter points."""
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    setup_ieee_style()

    YLIMS_BIAS = {
        0.8: {10: (-1.0, 1.5), 20: (-4, 1.5), 30: (-9, 2),
              40: (-6, 2), 50: (-2.5, 2), 60: (-2.5, 2)},
        0.2: {10: (-0.4, 0.4), 20: (-0.4, 1.0), 30: (-1.0, 1.2),
              40: (-0.8, 1.2), 50: (-0.7, 0.8), 60: (-1.0, 1.0)},
    }
    YFMT = {10: '%.1f', 20: '%.0f', 30: '%.0f',
            40: '%.0f', 50: '%.1f', 60: '%.1f'}

    for grams in LOADINGS_G:
        n_cols = len(DISTANCES_CM)
        fig, axes = plt.subplots(1, n_cols, figsize=(IEEE_DOUBLE_COL, 1.8))
        color = LOADING_COLORS[grams]

        for idx, d_cm in enumerate(DISTANCES_CM):
            ax = axes[idx]

            for trial in TRIALS:
                key = (grams, trial, d_cm)
                df = all_data.get(key)
                if df is None or len(df) == 0:
                    continue
                wall = df[~df['is_false_lock']].copy()
                if len(wall) == 0:
                    continue
                t_min = wall['time_seconds'].values / 60.0
                bias = wall['distance_cm'].values - d_cm
                ax.scatter(t_min, bias, s=0.3, alpha=0.15, color=color,
                           linewidths=0, rasterized=True)

            ax.axhline(0, color='0.4', ls='--', lw=0.5)
            ax.set_title(f'{d_cm} cm', fontsize=8)
            ylims_g = YLIMS_BIAS.get(grams, YLIMS_BIAS[0.8])
            if d_cm in ylims_g:
                ax.set_ylim(ylims_g[d_cm])
            ax.yaxis.set_major_locator(LinearLocator(numticks=5))
            if d_cm in YFMT:
                ax.yaxis.set_major_formatter(FormatStrFormatter(YFMT[d_cm]))
            ax.set_xlabel('Time (min)')
            if idx == 0:
                ax.set_ylabel('Bias (cm)')

        style_handles = [
            Line2D([], [], color=color, ls='', marker='o',
                   markersize=2, label=f'{grams} g'),
        ]
        fig.legend(handles=style_handles,
                   loc='lower center', ncol=1, fontsize=6,
                   bbox_to_anchor=(0.5, -0.03))

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        path = os.path.join(OUTPUT_DIR,
                            f'fig_a6_bias_recovery_raw_{grams}g.png')
        save_fig(fig, path)
        plt.close(fig)
        print(f"  Saved {path}")


def generate_plots(analysis_results):
    """Generate all model-vs-experiment comparison figures."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _dense_cache.clear()

    ts = analysis_results['timeseries_metrics']
    sc = analysis_results['scalar_metrics']
    all_data = analysis_results['all_data']

    print("\nGenerating comparison plots...")

    print("[1/6] Scatter overlay (0.8g, runs stochastic model)...")
    plot_scatter_overlay(all_data, sc)

    print("[2/6] False-lock fraction comparison...")
    plot_false_lock_comparison(ts, all_data=all_data)

    print("[3/6] Wall-locked bias recovery...")
    plot_bias_recovery(ts, all_data=all_data)

    print("[4/6] Wall-locked bias recovery (aligned zero)...")
    plot_bias_recovery_overlay_aligned(ts, all_data=all_data)

    print("[5/6] Wall-locked bias recovery (wide strip)...")
    plot_bias_recovery_wide(ts, all_data)

    print("[6/6] Wall-locked bias recovery (raw scatter)...")
    plot_bias_recovery_raw(all_data)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


__all__ = [
    'generate_plots',
    'plot_scatter_overlay',
    'plot_false_lock_comparison',
    'plot_bias_recovery',
    'plot_bias_recovery_overlay_aligned',
    'plot_bias_recovery_wide',
    'plot_bias_recovery_raw',
]


if __name__ == '__main__':
    from tof_analysis import run_analysis
    results = run_analysis()
    generate_plots(results)
