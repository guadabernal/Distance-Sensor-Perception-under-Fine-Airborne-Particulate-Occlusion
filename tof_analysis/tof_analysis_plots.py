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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dust_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'statistical_analysis'))
from plot_style import (
    setup_ieee_style, save_fig,
    IEEE_SINGLE_COL, IEEE_DOUBLE_COL,
    TOL_MUTED, DISTANCE_COLORS, TRIAL_COLORS, LOADING_COLORS,
    PANEL_ORDER, PANEL_LABELS, PANEL_YLIMS,
    sort_by_concentration, CONC_AXIS_LABEL, CONC_AXIS_LABEL_TWO_LINE,
    CONC_AXIS_LABEL_SHORT,
)
from matplotlib.patches import Patch

from tof_analysis import (
    run_model_stochastic, load_raw_center_readings, CENTER_CM_COLS,
    DISTANCES_CM, LOADINGS_G, TRIALS,
    FL_THRESHOLD_FRAC,
)
from dust_model import get_measurement_total_concentration_g_m3
from false_lock_mechanism import (
    run_aware_state_curve, interpolate_runs_to_common_grid,
)

DIST_COLORS = DISTANCE_COLORS
YLIMS = PANEL_YLIMS

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')

CONC_XLIMS = {0.2: (0.9, 2.35), 0.8: (3.8, 9.1)}
CONC_XTICKS = {0.2: [1.0, 1.6, 2.2], 0.8: [4.0, 6.5, 9.0]}
CONC_XLIMS_BOTH = (0.8, 9.2)
CONC_XTICKS_BOTH = [1, 5, 9]

_conc_cache = {}


def _conc_g_m3(grams, t_s):
    """Modeled chamber-average airborne concentration [g/m^3] at phase-local t_s."""
    t = np.asarray(t_s, dtype=float)
    key = (float(grams), t.tobytes())
    if key not in _conc_cache:
        _conc_cache[key] = get_measurement_total_concentration_g_m3(
            float(grams), t)
    return _conc_cache[key]


def _x(grams, t_min, x_axis, *arrays):
    """Experimental x-values (time [min] or modeled C_air), with companion arrays reordered to match."""
    if x_axis == 'conc':
        return sort_by_concentration(_conc_g_m3(grams, t_min * 60.0), *arrays)
    return (t_min,) + arrays


def _x_model(dense, x_axis, *arrays, valid=slice(None), step=1):
    """Dense-model x-values (time [min] or the model's C_air), with companion arrays reordered to match."""
    if x_axis == 'conc':
        return sort_by_concentration(dense['C_air_g_m3'][valid][::step], *arrays)
    return (dense['t_min'][valid][::step],) + arrays


SUFFIX = {'time': '', 'conc': '_vs_concentration'}


def plot_scatter_overlay(all_data, scalar_metrics, x_axis='time'):
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
            x_exp = t_s / 60.0 if x_axis == 'time' else _conc_g_m3(grams, t_s)
            for col_data in vals.values():
                ax.scatter(x_exp, col_data, s=0.5, alpha=0.35,
                           color=TRIAL_COLORS[trial_idx],
                           linewidths=0, zorder=2)

        print(f"    Running stochastic sim {grams}g @ {dist_cm}cm "
              f"({len(t_grid_s)} pts x {n_frames} frames)...")
        result = run_model_stochastic(dist_cm, grams, t_grid_s, n_frames=n_frames)
        x_sim = np.repeat(T_MIN if x_axis == 'time' else result['C_air_g_m3'],
                          n_frames)
        sim_flat = result['distances_cm'].flatten()
        valid = ~np.isnan(sim_flat)
        ax.scatter(x_sim[valid], sim_flat[valid],
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
                x_snap = (snap_min if x_axis == 'time' else
                          _conc_g_m3(grams, np.array([snap_min * 60.0]))[0])
                ax.axvline(x_snap, color='#e06000', ls='--', lw=0.8,
                           alpha=0.7, zorder=5)

        y_lo, y_hi = YLIMS[dist_cm]
        ax.set_ylim(y_lo, y_hi)
        if x_axis == 'conc':
            ax.set_xlim(CONC_XLIMS[grams])
            ax.set_xlabel(CONC_AXIS_LABEL, fontsize=9)
        else:
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
               label='Model snap time' if x_axis == 'time'
               else 'Model snap concentration'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=6,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))

    from tof_model import ALPHA_0_MEASURED, S_LIDAR as _S
    x_note = ('' if x_axis == 'time' else
              'x-axis: model-estimated chamber-average airborne concentration; ')
    fig.suptitle(
        f'Model vs Experiment — {grams} g Arizona ISO A0\n'
        f'{x_note}ALPHA_0 = {ALPHA_0_MEASURED}, S_LIDAR = {_S} sr',
        fontsize=10, y=1.01)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'fig_a1_scatter{SUFFIX[x_axis]}.png')
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
        'C_air_g_m3': result['C_air_g_m3'],
    }


_dense_cache = {}


def _get_dense(dist_cm, mass_g):
    key = (dist_cm, mass_g)
    if key not in _dense_cache:
        _dense_cache[key] = _dense_model_curves(dist_cm, mass_g)
    return _dense_cache[key]


def plot_false_lock_comparison(ts_metrics, all_data=None, x_axis='time'):
    """Fig 2: f_FL(t) curves, exp vs model, 0.8g only."""
    setup_ieee_style()

    grams = 0.8
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.5))
    plotted_dists = []
    if x_axis == 'conc':
        x_of_t = lambda t: _conc_g_m3(grams, t)
    else:
        x_of_t = lambda t: t / 60.0

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
                t_s, vals = load_raw_center_readings(grams, dist_cm, trial)
                if t_s is None or len(vals) < len(CENTER_CM_COLS):
                    continue
                cm = np.column_stack([vals[c] for c in CENTER_CM_COLS])
                _, fl_smooth, x_run = run_aware_state_curve(
                    t_s, cm, dist_cm, x_of_t, smooth_window_s=30.0)
                trial_curves.append((x_run, fl_smooth))

            if len(trial_curves) > 1:
                x_lo = max(x.min() for x, _ in trial_curves)
                x_hi = min(x.max() for x, _ in trial_curves)
                common_x = np.linspace(x_lo, x_hi, 300)
                _, mean_fl = interpolate_runs_to_common_grid(
                    trial_curves, common_x)
                ax.plot(common_x, mean_fl, color=color, ls='-', lw=1.0)
            elif len(trial_curves) == 1:
                x_run, fl = trial_curves[0]
                if x_axis == 'conc':
                    x_run, fl = sort_by_concentration(x_run, fl)
                ax.plot(x_run, fl, color=color, ls='-', lw=1.0)

        ax.plot(*_x_model(dense, x_axis, dense['f_fl']),
                color=color, ls='--', lw=1.0)

    ax.set_ylim(-0.02, 1.02)
    if x_axis == 'conc':
        c_win_lo = _conc_g_m3(grams, np.array([5.0 * 60.0]))[0]
        c_win_hi = _conc_g_m3(grams, np.array([0.0]))[0]
        pad = 0.02 * (c_win_hi - c_win_lo)
        ax.set_xlim(c_win_lo - pad, c_win_hi + pad)
        ax.set_xlabel(CONC_AXIS_LABEL_TWO_LINE)
    else:
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
    ax.legend(handles=legend_handles, fontsize=6,
              loc='upper right' if x_axis == 'time' else 'upper left')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'fig_a2_false_lock{SUFFIX[x_axis]}.png')
    save_fig(fig, path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_bias_recovery(ts_metrics, all_data=None, x_axis='time'):
    """Fig 3: 2x6 grid — wall-locked % error vs time (rows=loadings, cols=distances)."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    setup_ieee_style()

    n_cols = len(DISTANCES_CM)
    fig, axes = plt.subplots(2, n_cols, figsize=(IEEE_DOUBLE_COL, 3.2),
                             sharex=(x_axis == 'time'))

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
                    std_err = stacked.std(axis=1, ddof=1)
                    x, mean_err, std_err = _x(grams, common_t, x_axis,
                                              mean_err, std_err)
                    ax.plot(x, mean_err, color=color, linewidth=0.8)
                    ax.fill_between(x, mean_err - std_err,
                                    mean_err + std_err, color=color, alpha=0.15)
                elif len(trial_curves) == 1:
                    tc, err = trial_curves[0]
                    ax.plot(*_x(grams, tc, x_axis, err), color=color, linewidth=0.8)

            dense = _get_dense(dist_cm, grams)
            valid_d = ~np.isnan(dense['bias_pct_smooth'])
            if valid_d.any():
                m_smooth = dense['bias_pct_smooth'][valid_d]
                m_std = dense['bias_pct_std'][valid_d]
                step = 7
                ax.plot(*_x_model(dense, x_axis, m_smooth[::step],
                                  valid=valid_d, step=step),
                        color=MODEL_COLOR, linewidth=0.8, ls='--', zorder=10)
                ax.fill_between(*_x_model(dense, x_axis, m_smooth - m_std,
                                          m_smooth + m_std, valid=valid_d),
                                color=MODEL_COLOR, alpha=0.12, zorder=9)

            ax.axhline(0, color='0.4', ls='--', lw=0.5)
            if dist_cm in ylims_g:
                ax.set_ylim(ylims_g[dist_cm])
            ax.yaxis.set_major_locator(LinearLocator(numticks=5))
            if dist_cm in YFMT:
                ax.yaxis.set_major_formatter(FormatStrFormatter(YFMT[dist_cm]))

            if x_axis == 'conc':
                ax.set_xlim(CONC_XLIMS[grams])
                ax.set_xticks(CONC_XTICKS[grams])
            if row_idx == 0:
                ax.set_title(f'{dist_cm} cm', fontsize=8)
            if x_axis == 'conc':
                ax.set_xlabel(CONC_AXIS_LABEL_SHORT)
            elif row_idx == 1:
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
    path = os.path.join(OUTPUT_DIR, f'fig_a3_bias_recovery{SUFFIX[x_axis]}.png')
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
                    std_err = stacked.std(axis=1, ddof=1)
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


def plot_bias_recovery_overlay_aligned_conc(ts_metrics, all_data=None):
    """Fig 3d (concentration axis): 2x6 grid (rows=loadings) with zero aligned per loading."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import FormatStrFormatter
    setup_ieee_style()

    n_rows = len(LOADINGS_G)
    n_cols = len(DISTANCES_CM)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(IEEE_DOUBLE_COL, 3.4))

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

    CONC_DATA_XLIMS = {
        g: (float(_conc_g_m3(g, np.array([900.0]))[0]),
            float(_conc_g_m3(g, np.array([0.0]))[0]))
        for g in LOADINGS_G
    }
    CONC_DATA_XTICKS = {0.2: [1.0, 1.6, 2.2], 0.8: [4.0, 6.0, 8.0]}

    panel_yvals = {(g, d): [] for g in LOADINGS_G for d in DISTANCES_CM}

    for row_idx, grams in enumerate(LOADINGS_G):
        exp_color = EXP_COLORS[grams]
        mod_color = MODEL_COLORS[grams]

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
                    std_err = stacked.std(axis=1, ddof=1)
                    c_exp, mean_err, std_err = sort_by_concentration(
                        _conc_g_m3(grams, common_t * 60.0), mean_err, std_err)
                    ax.plot(c_exp, mean_err, color=exp_color, linewidth=0.9)
                    ax.fill_between(c_exp, mean_err - std_err,
                                    mean_err + std_err, color=exp_color, alpha=0.18)
                    panel_yvals[(grams, dist_cm)].extend(mean_err + std_err)
                    panel_yvals[(grams, dist_cm)].extend(mean_err - std_err)
                elif len(trial_curves) == 1:
                    tc, err = trial_curves[0]
                    c_exp, err_c = sort_by_concentration(
                        _conc_g_m3(grams, tc * 60.0), err)
                    ax.plot(c_exp, err_c, color=exp_color, linewidth=0.9)
                    panel_yvals[(grams, dist_cm)].extend(err_c)

            dense = _get_dense(dist_cm, grams)
            valid_d = ~np.isnan(dense['bias_pct_smooth'])
            if valid_d.any():
                c_valid = dense['C_air_g_m3'][valid_d]
                m_smooth = dense['bias_pct_smooth'][valid_d]
                step = 7
                c_step, m_step = sort_by_concentration(
                    c_valid[::step], m_smooth[::step])
                ax.plot(c_step, m_step, color=mod_color,
                        linewidth=0.9, ls='--', zorder=10)
                panel_yvals[(grams, dist_cm)].extend(m_smooth)

            ax.axhline(0, color='0.4', ls='--', lw=0.5)
            if row_idx == 0:
                ax.set_title(f'{dist_cm} cm', fontsize=8)
            ax.set_xlabel(CONC_AXIS_LABEL_SHORT)
            if col_idx == 0:
                ax.set_ylabel(f'{grams} g\nError (%)')

    Y_TOP_FIXED = {50: 0.3, 60: 0.3}
    ZERO_RATIO = {0.2: 1.0, 0.8: 3.0}

    for row_idx, grams in enumerate(LOADINGS_G):
        ratio = ZERO_RATIO[grams]
        for col_idx, dist_cm in enumerate(DISTANCES_CM):
            ax = axes[row_idx, col_idx]
            vals = np.array(panel_yvals[(grams, dist_cm)])
            if len(vals) == 0:
                continue
            if dist_cm in Y_TOP_FIXED:
                y_top = Y_TOP_FIXED[dist_cm]
            else:
                data_max = np.nanmax(vals) * PAD
                data_min = np.nanmin(vals) * PAD
                y_top_from_max = max(data_max, 0.01)
                y_top_from_min = abs(data_min) / ratio if data_min < 0 else 0.01
                y_top = float(np.ceil(max(y_top_from_max, y_top_from_min)))
            y_bot = -ratio * y_top
            ax.set_ylim(y_bot, y_top)
            ax.set_xlim(CONC_DATA_XLIMS[grams])
            ax.xaxis.set_major_locator(plt.FixedLocator(CONC_DATA_XTICKS[grams]))
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
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.02, 1, 1], w_pad=0.15, h_pad=1.1)
    path = os.path.join(OUTPUT_DIR,
                        'fig_a4_bias_recovery_aligned_vs_concentration.png')
    save_fig(fig, path)
    plt.close(fig)
    print(f"  Saved {path}")


LOADING_COLORS = {0.2: TOL_MUTED['indigo'], 0.8: TOL_MUTED['rose']}


def plot_bias_recovery_wide(ts_metrics, all_data, x_axis='time'):
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
                ax.plot(*_x(grams, t_c, x_axis, err),
                        color=color, linewidth=0.4, alpha=0.25)

            if len(trial_curves) > 1:
                t_lo = max(tc.min() for tc, _ in trial_curves)
                t_hi = min(tc.max() for tc, _ in trial_curves)
                common_t = np.linspace(t_lo, t_hi, 300)
                stacked = np.column_stack([
                    np.interp(common_t, tc, err)
                    for tc, err in trial_curves
                ])
                mean_err = stacked.mean(axis=1)
                std_err = stacked.std(axis=1, ddof=1)
                x, mean_err, std_err = _x(grams, common_t, x_axis,
                                          mean_err, std_err)
                ax.plot(x, mean_err, color=color, linewidth=0.8)
                ax.fill_between(x, mean_err - std_err,
                                mean_err + std_err, color=color, alpha=0.15)
            elif len(trial_curves) == 1:
                tc, err = trial_curves[0]
                ax.plot(*_x(grams, tc, x_axis, err), color=color, linewidth=0.8)

            dense = _get_dense(d_cm, grams)
            valid_d = ~np.isnan(dense['bias_pct_smooth'])
            if valid_d.any():
                m_smooth = dense['bias_pct_smooth'][valid_d]
                m_std = dense['bias_pct_std'][valid_d]
                x_m, m_smooth, m_lo, m_hi = _x_model(
                    dense, x_axis, m_smooth, m_smooth - m_std,
                    m_smooth + m_std, valid=valid_d)
                ax.plot(x_m, m_smooth, color=MODEL_COLOR, linewidth=0.8,
                        ls='--', zorder=10)
                ax.fill_between(x_m, m_lo, m_hi,
                                color=MODEL_COLOR, alpha=0.12, zorder=9)

            ax.axhline(0, color='0.4', ls='--', lw=0.5)
            ylims_g = YLIMS_PCT.get(grams, YLIMS_PCT[0.8])
            if d_cm in ylims_g:
                ax.set_ylim(ylims_g[d_cm])
            ax.yaxis.set_major_locator(LinearLocator(numticks=5))
            if d_cm in YFMT:
                ax.yaxis.set_major_formatter(FormatStrFormatter(YFMT[d_cm]))

            if x_axis == 'conc':
                ax.set_xlim(CONC_XLIMS[grams])
                ax.set_xticks(CONC_XTICKS[grams])
            ax.set_title(f'{d_cm} cm', fontsize=8)
            ax.set_xlabel('Time (min)' if x_axis == 'time' else CONC_AXIS_LABEL_SHORT)
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
        path = os.path.join(
            OUTPUT_DIR, f'fig_a5_bias_recovery_wide{SUFFIX[x_axis]}_{grams}g.png')
        save_fig(fig, path)
        plt.close(fig)
        print(f"  Saved {path}")


def plot_bias_recovery_raw(all_data, x_axis='time'):
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
                t_s = wall['time_seconds'].values
                x = t_s / 60.0 if x_axis == 'time' else _conc_g_m3(grams, t_s)
                bias = wall['distance_cm'].values - d_cm
                ax.scatter(x, bias, s=0.3, alpha=0.15, color=color,
                           linewidths=0, rasterized=True)

            ax.axhline(0, color='0.4', ls='--', lw=0.5)
            ax.set_title(f'{d_cm} cm', fontsize=8)
            ylims_g = YLIMS_BIAS.get(grams, YLIMS_BIAS[0.8])
            if d_cm in ylims_g:
                ax.set_ylim(ylims_g[d_cm])
            ax.yaxis.set_major_locator(LinearLocator(numticks=5))
            if d_cm in YFMT:
                ax.yaxis.set_major_formatter(FormatStrFormatter(YFMT[d_cm]))
            if x_axis == 'conc':
                ax.set_xlim(CONC_XLIMS[grams])
                ax.set_xticks(CONC_XTICKS[grams])
            ax.set_xlabel('Time (min)' if x_axis == 'time' else CONC_AXIS_LABEL_SHORT)
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
        path = os.path.join(
            OUTPUT_DIR, f'fig_a6_bias_recovery_raw{SUFFIX[x_axis]}_{grams}g.png')
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

    print("[1/12] Scatter overlay (0.8g, runs stochastic model)...")
    plot_scatter_overlay(all_data, sc)

    print("[2/12] Scatter overlay vs concentration...")
    plot_scatter_overlay(all_data, sc, x_axis='conc')

    print("[3/12] False-lock fraction comparison...")
    plot_false_lock_comparison(ts, all_data=all_data)

    print("[4/12] False-lock fraction vs concentration...")
    plot_false_lock_comparison(ts, all_data=all_data, x_axis='conc')

    print("[5/12] Wall-locked bias recovery...")
    plot_bias_recovery(ts, all_data=all_data)

    print("[6/12] Wall-locked bias recovery vs concentration...")
    plot_bias_recovery(ts, all_data=all_data, x_axis='conc')

    print("[7/12] Wall-locked bias recovery (aligned zero)...")
    plot_bias_recovery_overlay_aligned(ts, all_data=all_data)

    print("[8/12] Wall-locked bias recovery (aligned) vs concentration...")
    plot_bias_recovery_overlay_aligned_conc(ts, all_data=all_data)

    print("[9/12] Wall-locked bias recovery (wide strip)...")
    plot_bias_recovery_wide(ts, all_data)

    print("[10/12] Wall-locked bias recovery (wide) vs concentration...")
    plot_bias_recovery_wide(ts, all_data, x_axis='conc')

    print("[11/12] Wall-locked bias recovery (raw scatter)...")
    plot_bias_recovery_raw(all_data)

    print("[12/12] Wall-locked bias recovery (raw) vs concentration...")
    plot_bias_recovery_raw(all_data, x_axis='conc')

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


__all__ = [
    'generate_plots',
    'plot_scatter_overlay',
    'plot_false_lock_comparison',
    'plot_bias_recovery',
    'plot_bias_recovery_overlay_aligned',
    'plot_bias_recovery_overlay_aligned_conc',
    'plot_bias_recovery_wide',
    'plot_bias_recovery_raw',
]


if __name__ == '__main__':
    from tof_analysis import run_analysis
    results = run_analysis()
    generate_plots(results)
