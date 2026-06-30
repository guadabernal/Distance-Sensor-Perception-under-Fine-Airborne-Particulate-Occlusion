"""Generate IR sensor analysis figures, metrics CSV, ANOVA, and paper summary."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import (
    setup_ieee_style, save_fig,
    IEEE_SINGLE_COL, IEEE_DOUBLE_COL,
    TOL_MUTED, LOADING_COLORS, DISTANCE_COLORS,
)

from ir_analysis import (
    load_trial1_data, load_trial_data, block_average,
    multi_trial_block_average,
    compute_all_metrics, compute_anova, fit_exponential_decay,
    fit_error_vs_distance, save_paper_summary, phase_time_seconds,
    DISTANCES_CM, DUST_LOADINGS_G, TRIALS,
)
from ir_model import simulate_experiment


def _fmt(val, decimals=2):
    """Format a metric value for table display."""
    if isinstance(val, str):
        return val
    if not np.isfinite(val):
        return 'N/R'
    if abs(val) >= 1000:
        return f'{val:.0f}'
    if abs(val) >= 10:
        return f'{val:.{min(decimals, 1)}f}'
    return f'{val:.{decimals}f}'


def plot_measured_vs_actual(control_data, save_path):
    """Fig 1: scatter of control-phase mean distance vs actual distance."""
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.6))

    style = {0.2: ('o', '0.2 g'), 0.8: ('s', '0.8 g')}

    lims = [0, max(DISTANCES_CM) + 5]
    ax.plot(lims, lims, ls='--', color='0.5', lw=0.8, label='Ideal ($y = x$)')

    for grams in DUST_LOADINGS_G:
        true_dists, mean_dists, std_dists = control_data[grams]
        marker, label = style[grams]
        ax.errorbar(
            true_dists, mean_dists, yerr=std_dists,
            fmt=marker, color=LOADING_COLORS[grams], markersize=4,
            capsize=3, elinewidth=0.8, label=label,
        )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Actual distance (cm)')
    ax.set_ylabel('Measured distance (cm)')
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_error_comparison_grid(error_comp_data, save_path):
    """Fig 2: 5x2 grid (rows=distances, cols=loadings) — trials, mean+/-SD, and model."""
    setup_ieee_style()
    n_dists = len(DISTANCES_CM)
    n_loadings = len(DUST_LOADINGS_G)
    fig, axes = plt.subplots(
        n_dists, n_loadings,
        figsize=(IEEE_SINGLE_COL, n_dists * 1.1 + 0.6))

    for col, grams in enumerate(DUST_LOADINGS_G):
        for row, (d_cm, trial_errors,
                  model_t_min, model_err) in enumerate(error_comp_data[grams]):
            ax = axes[row, col]
            color = DISTANCE_COLORS[d_cm]

            for t_min, err in trial_errors:
                ax.plot(t_min, err, color=color, linewidth=0.5, alpha=0.35)

            if len(trial_errors) > 1:
                min_len = min(len(e) for _, e in trial_errors)
                stacked = np.column_stack([e[:min_len] for _, e in trial_errors])
                t_common = trial_errors[0][0][:min_len]
                err_mean = stacked.mean(axis=1)
                err_std = stacked.std(axis=1)
                ax.plot(t_common, err_mean, color=color, linewidth=1.0)
                ax.fill_between(t_common, err_mean - err_std,
                                err_mean + err_std,
                                color=color, alpha=0.18)

            ax.plot(model_t_min, model_err, color=color, linewidth=0.9, ls='--')
            ax.axhline(0, color='0.4', ls='--', lw=0.6)

            if row == 0:
                ax.set_title(f'{grams} g')
            if row == n_dists - 1:
                ax.set_xlabel('Time (min)')
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(f'{d_cm} cm\nError (cm)')
            else:
                ax.tick_params(labelleft=False)

    from matplotlib.ticker import FormatStrFormatter
    for row in range(n_dists):
        ylims = [axes[row, col].get_ylim() for col in range(n_loadings)]
        ymin = min(y[0] for y in ylims)
        ymax = max(y[1] for y in ylims)
        for col in range(n_loadings):
            axes[row, col].set_ylim(ymin, ymax)
            axes[row, col].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    style_handles = [
        Line2D([], [], color='0.3', ls='-', lw=0.5, alpha=0.35,
               label='Trial'),
        Line2D([], [], color='0.3', ls='-', lw=1.0, label='Mean'),
        Patch(facecolor='0.7', alpha=0.18, label='$\\pm$1 SD'),
        Line2D([], [], color='0.3', ls='--', lw=0.9, label='Model'),
    ]
    fig.legend(handles=style_handles,
               loc='lower center', ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, save_path)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_error_comparison_grid_pct(error_comp_data, save_path):
    """Fig 2v2: same grid as fig 2 with percent error on the y-axis."""
    setup_ieee_style()
    n_dists = len(DISTANCES_CM)
    n_loadings = len(DUST_LOADINGS_G)
    fig, axes = plt.subplots(
        n_dists, n_loadings,
        figsize=(IEEE_SINGLE_COL, n_dists * 1.1 + 0.6))

    for col, grams in enumerate(DUST_LOADINGS_G):
        for row, (d_cm, trial_errors,
                  model_t_min, model_err) in enumerate(error_comp_data[grams]):
            ax = axes[row, col]
            color = DISTANCE_COLORS[d_cm]

            trial_errors_pct = []
            for t_min, err in trial_errors:
                err_pct = err / d_cm * 100.0
                ax.plot(t_min, err_pct, color=color, linewidth=0.5, alpha=0.35)
                trial_errors_pct.append((t_min, err_pct))

            if len(trial_errors_pct) > 1:
                min_len = min(len(e) for _, e in trial_errors_pct)
                stacked = np.column_stack(
                    [e[:min_len] for _, e in trial_errors_pct])
                t_common = trial_errors_pct[0][0][:min_len]
                err_mean = stacked.mean(axis=1)
                err_std = stacked.std(axis=1)
                ax.plot(t_common, err_mean, color=color, linewidth=1.0)
                ax.fill_between(t_common, err_mean - err_std,
                                err_mean + err_std,
                                color=color, alpha=0.18)

            model_err_pct = model_err / d_cm * 100.0
            ax.plot(model_t_min, model_err_pct, color=color,
                    linewidth=0.9, ls='--')
            ax.axhline(0, color='0.4', ls='--', lw=0.6)

            if row == 0:
                ax.set_title(f'{grams} g')
            if row == n_dists - 1:
                ax.set_xlabel('Time (min)')
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(f'{d_cm} cm\nError (%)')
            else:
                ax.tick_params(labelleft=False)

    from matplotlib.ticker import FormatStrFormatter
    for row in range(n_dists):
        ylims = [axes[row, col].get_ylim() for col in range(n_loadings)]
        ymin = min(y[0] for y in ylims)
        ymax = max(y[1] for y in ylims)
        for col in range(n_loadings):
            axes[row, col].set_ylim(ymin, ymax)
            axes[row, col].yaxis.set_major_formatter(
                FormatStrFormatter('%.1f'))

    style_handles = [
        Line2D([], [], color='0.3', ls='-', lw=0.5, alpha=0.35,
               label='Trial'),
        Line2D([], [], color='0.3', ls='-', lw=1.0, label='Mean'),
        Patch(facecolor='0.7', alpha=0.18, label='$\\pm$1 SD'),
        Line2D([], [], color='0.3', ls='--', lw=0.9, label='Model'),
    ]
    fig.legend(handles=style_handles,
               loc='lower center', ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, save_path)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_error_comparison_overlay_pct(error_comp_data, save_path):
    """Fig 2v3: 2x3 grid (one panel per distance), both loadings overlaid."""
    setup_ieee_style()
    GRID_DISTANCES = [10, 20, 30, 40, 50, 60]
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(IEEE_DOUBLE_COL, n_rows * 1.3 + 0.6))

    from matplotlib.ticker import MaxNLocator, LinearLocator, FormatStrFormatter
    YLIMS = {10: (-1, 1), 20: (-5, 0), 30: (-10, 0), 40: (-25, 0), 50: (-40, 0), 60: (-50, 0)}
    YFMT = {10: '%.1f', 20: '%.1f', 30: '%.1f', 40: '%.0f', 50: '%.0f', 60: '%.0f'}

    for idx, d_cm in enumerate(GRID_DISTANCES):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        has_data = d_cm in DISTANCES_CM

        if has_data:
            for grams in DUST_LOADINGS_G:
                entry = [e for e in error_comp_data[grams] if e[0] == d_cm][0]
                _, trial_errors, model_t_min, model_err = entry
                color = LOADING_COLORS[grams]

                for t_min, err in trial_errors:
                    ax.plot(t_min, err / d_cm * 100.0,
                            color=color, linewidth=0.5, alpha=0.25)

                if len(trial_errors) > 1:
                    min_len = min(len(e) for _, e in trial_errors)
                    stacked = np.column_stack(
                        [e[:min_len] / d_cm * 100.0 for _, e in trial_errors])
                    t_common = trial_errors[0][0][:min_len]
                    err_mean = stacked.mean(axis=1)
                    err_std = stacked.std(axis=1)
                    ax.plot(t_common, err_mean, color=color, linewidth=1.0,
                            label=f'{grams} g')
                    ax.fill_between(t_common, err_mean - err_std,
                                    err_mean + err_std, color=color, alpha=0.15)

                ax.plot(model_t_min, model_err / d_cm * 100.0,
                        color=color, linewidth=0.9, ls='--')
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='0.5')

        ax.axhline(0, color='0.4', ls='--', lw=0.6)
        ax.set_title(f'{d_cm} cm', fontsize=8)
        if d_cm in YLIMS:
            ax.set_ylim(YLIMS[d_cm])
        ax.yaxis.set_major_locator(LinearLocator(numticks=5))
        if d_cm in YFMT:
            ax.yaxis.set_major_formatter(FormatStrFormatter(YFMT[d_cm]))

        if row == n_rows - 1 or idx == len(GRID_DISTANCES) - 1:
            ax.set_xlabel('Time (min)')
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel('Error (%)')

        if idx == 0:
            ax.legend(fontsize=6)

    for idx in range(len(GRID_DISTANCES), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    style_handles = [
        Line2D([], [], color='0.3', ls='-', lw=0.5, alpha=0.25,
               label='Trial'),
        Line2D([], [], color='0.3', ls='-', lw=1.0, label='Mean'),
        Patch(facecolor='0.7', alpha=0.15, label='$\\pm$1 SD'),
        Line2D([], [], color='0.3', ls='--', lw=0.9, label='Model'),
    ]
    fig.legend(handles=style_handles,
               loc='lower center', ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, save_path)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_error_comparison_overlay_pct_wide(error_comp_data, save_path):
    """Fig 2v4: 1x6 wide strip for IEEE double-column width."""
    setup_ieee_style()
    GRID_DISTANCES = [10, 20, 30, 40, 50, 60]
    n_cols = len(GRID_DISTANCES)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(IEEE_DOUBLE_COL, 1.8))

    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    YLIMS = {10: (-1, 1), 20: (-5, 0), 30: (-10, 0),
             40: (-25, 0), 50: (-45, 0), 60: (-65, 0)}
    YFMT = {10: '%.1f', 20: '%.1f', 30: '%.1f',
            40: '%.0f', 50: '%.0f', 60: '%.0f'}

    for idx, d_cm in enumerate(GRID_DISTANCES):
        ax = axes[idx]
        has_data = d_cm in DISTANCES_CM

        if has_data:
            for grams in DUST_LOADINGS_G:
                entry = [e for e in error_comp_data[grams] if e[0] == d_cm][0]
                _, trial_errors, model_t_min, model_err = entry
                color = LOADING_COLORS[grams]

                for t_min, err in trial_errors:
                    ax.plot(t_min, err / d_cm * 100.0,
                            color=color, linewidth=0.4, alpha=0.25)

                if len(trial_errors) > 1:
                    min_len = min(len(e) for _, e in trial_errors)
                    stacked = np.column_stack(
                        [e[:min_len] / d_cm * 100.0 for _, e in trial_errors])
                    t_common = trial_errors[0][0][:min_len]
                    err_mean = stacked.mean(axis=1)
                    err_std = stacked.std(axis=1)
                    ax.plot(t_common, err_mean, color=color, linewidth=0.8,
                            label=f'{grams} g')
                    ax.fill_between(t_common, err_mean - err_std,
                                    err_mean + err_std, color=color, alpha=0.15)

                ax.plot(model_t_min, model_err / d_cm * 100.0,
                        color=color, linewidth=0.7, ls='--')
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=7, color='0.5')

        ax.axhline(0, color='0.4', ls='--', lw=0.5)
        ax.set_title(f'{d_cm} cm', fontsize=8)
        if d_cm in YLIMS:
            ax.set_ylim(YLIMS[d_cm])
        ax.yaxis.set_major_locator(LinearLocator(numticks=5))
        if d_cm in YFMT:
            ax.yaxis.set_major_formatter(FormatStrFormatter(YFMT[d_cm]))

        ax.set_xlabel('Time (min)')
        if idx == 0:
            ax.set_ylabel('Error (%)')

    style_handles = [
        Line2D([], [], color=LOADING_COLORS[0.2], ls='-', lw=0.8,
               label='0.2 g'),
        Line2D([], [], color=LOADING_COLORS[0.8], ls='-', lw=0.8,
               label='0.8 g'),
        Line2D([], [], color='0.3', ls='-', lw=0.4, alpha=0.25,
               label='Trial'),
        Line2D([], [], color='0.3', ls='-', lw=0.8, label='Mean'),
        Patch(facecolor='0.7', alpha=0.15, label='$\\pm$1 SD'),
        Line2D([], [], color='0.3', ls='--', lw=0.7, label='Model'),
    ]
    fig.legend(handles=style_handles,
               loc='lower center', ncol=6, fontsize=6,
               bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, save_path)
    plt.close(fig)


def plot_error_comparison_overlay_pct_v5(error_comp_data, save_path):
    """Fig 2v5: 1x6 wide strip — no trial lines, legend shows gram mean/std."""
    setup_ieee_style()
    GRID_DISTANCES = [10, 20, 30, 40, 50, 60]
    n_cols = len(GRID_DISTANCES)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(IEEE_DOUBLE_COL, 1.8))

    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    YLIMS = {10: (-1, 1), 20: (-5, 0), 30: (-10, 0),
             40: (-25, 0), 50: (-45, 0), 60: (-65, 0)}
    YFMT = {10: '%.1f', 20: '%.1f', 30: '%.1f',
            40: '%.0f', 50: '%.0f', 60: '%.0f'}

    def _lighten(hex_color, amount=0.4):
        h = hex_color.lstrip('#')
        r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = int(r + (255 - r) * amount)
        g = int(g + (255 - g) * amount)
        b = int(b + (255 - b) * amount)
        return f'#{r:02x}{g:02x}{b:02x}'

    EXP_COLORS = {g: _lighten(LOADING_COLORS[g]) for g in DUST_LOADINGS_G}
    MODEL_COLORS = LOADING_COLORS

    for idx, d_cm in enumerate(GRID_DISTANCES):
        ax = axes[idx]
        has_data = d_cm in DISTANCES_CM

        if has_data:
            for grams in DUST_LOADINGS_G:
                entry = [e for e in error_comp_data[grams] if e[0] == d_cm][0]
                _, trial_errors, model_t_min, model_err = entry
                exp_color = EXP_COLORS[grams]
                mod_color = MODEL_COLORS[grams]

                if len(trial_errors) > 1:
                    min_len = min(len(e) for _, e in trial_errors)
                    stacked = np.column_stack(
                        [e[:min_len] / d_cm * 100.0 for _, e in trial_errors])
                    t_common = trial_errors[0][0][:min_len]
                    err_mean = stacked.mean(axis=1)
                    err_std = stacked.std(axis=1)
                    ax.fill_between(t_common, err_mean - err_std,
                                    err_mean + err_std, color=exp_color, alpha=0.18)
                    ax.plot(t_common, err_mean, color=exp_color, linewidth=0.9)

                ax.plot(model_t_min, model_err / d_cm * 100.0,
                        color=mod_color, linewidth=0.9, ls='--')
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=7, color='0.5')

        ax.axhline(0, color='0.4', ls='--', lw=0.5)
        ax.set_title(f'{d_cm} cm', fontsize=8)
        if d_cm in YLIMS:
            ax.set_ylim(YLIMS[d_cm])
        ax.yaxis.set_major_locator(LinearLocator(numticks=5))
        if d_cm in YFMT:
            ax.yaxis.set_major_formatter(FormatStrFormatter(YFMT[d_cm]))

        ax.set_xlabel('Time (min)')
        ax.set_xticks([0, 5, 10, 15])
        if idx == 0:
            ax.set_ylabel('Error (%)')

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
    save_fig(fig, save_path)
    plt.close(fig)


def save_summary_table(df_metrics, save_path):
    """Save summary table as CSV with key columns."""
    cols = {
        'grams': 'Loading (g)',
        'distance_cm': 'Dist (cm)',
        'bias_full': 'Bias (cm)',
        'rmse_full': 'RMSE (cm)',
        'mae_full': 'MAE (cm)',
        'peak_abs_error': 'Peak (cm)',
        'peak_error_pct': 'Peak (%)',
        'time_to_peak_error_s': 't peak (s)',
        'recovery_time_5pct_s': 'Recov 5% (s)',
        'iae': 'IAE',
        'r_squared': 'R^2',
        'residual_rmse': 'Res RMSE',
        'correlation': 'Corr',
    }
    df = df_metrics[list(cols.keys())].rename(columns=cols)
    df = df.sort_values(['Loading (g)', 'Dist (cm)'])
    df.to_csv(save_path, index=False, float_format='%.3f')
    print(f'Saved: {save_path}')


def plot_distance_scaling(df_metrics, save_path, power_fits=None):
    """Fig 3: MAE vs distance for early (60s) and late (120s) windows."""
    setup_ieee_style()
    fig, (ax_e, ax_l) = plt.subplots(
        2, 1, figsize=(IEEE_SINGLE_COL, 4.8))

    d_fit = np.linspace(min(DISTANCES_CM), max(DISTANCES_CM), 100)

    for idx, (ax, col, title) in enumerate([
        (ax_e, 'early_mae_60s', 'Early (first 60 s)'),
        (ax_l, 'late_mae_120s', 'Late (last 120 s)'),
    ]):
        for grams, marker in [(0.2, 'o'), (0.8, 's')]:
            sub = df_metrics[df_metrics['grams'] == grams].sort_values(
                'distance_cm')
            ax.plot(
                sub['distance_cm'], sub[col],
                marker=marker, color=LOADING_COLORS[grams],
                markersize=4, linewidth=0.9, label=f'{grams} g',
            )

            if power_fits and (grams, col) in power_fits:
                fit = power_fits[(grams, col)]
                if not np.isnan(fit['n']):
                    y_fit = fit['a'] * d_fit ** fit['n']
                    ax.plot(d_fit, y_fit, color=LOADING_COLORS[grams],
                            ls='--', lw=0.7)
                    y_pos = 0.50 if grams == 0.2 else 0.35
                    ax.annotate(
                        f"$\\propto d^{{{fit['n']:.2f}}}$"
                        f" ($R^2$={fit['r_squared']:.2f})",
                        xy=(0.98, y_pos), xycoords='axes fraction',
                        fontsize=5, ha='right',
                        color=LOADING_COLORS[grams])

        if idx == 1:
            ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('MAE (cm)')
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_exponential_decay_fits(decay_data, save_path):
    """Fig 4: 1x2 — error curves with exponential decay fit overlay."""
    setup_ieee_style()
    fig, (ax_02, ax_08) = plt.subplots(
        1, 2, figsize=(IEEE_DOUBLE_COL, 2.8))

    for ax, grams, title in [(ax_02, 0.2, '0.2 g'), (ax_08, 0.8, '0.8 g')]:
        for entry in decay_data[grams]:
            d_cm, t_min, error, fit_curve, tau = entry[:5]
            model_err = entry[5] if len(entry) > 5 else None
            color = DISTANCE_COLORS[d_cm]
            ax.plot(t_min, error, color=color, linewidth=0.9,
                    label=f'{d_cm} cm')
            if fit_curve is not None:
                ax.plot(t_min, fit_curve, color=color, linewidth=0.7,
                        ls='--')
            if model_err is not None:
                ax.plot(t_min, model_err, color=color, linewidth=0.7,
                        ls=':')

        tau_text = '\n'.join(
            f'{e[0]} cm: $\\tau$={e[4]:.0f}s'
            for e in decay_data[grams]
            if not np.isnan(e[4])
        )
        if tau_text:
            ax.text(0.97, 0.03, tau_text, transform=ax.transAxes,
                    fontsize=5, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        ax.axhline(0, color='0.4', ls='--', lw=0.6)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Distance error (cm)')
        ax.set_title(title)
        ax.legend(title='True dist.', fontsize=6, title_fontsize=7,
                  loc='upper right')

    style_handles = [
        Line2D([], [], color='0.3', ls='-', lw=0.9, label='Data'),
        Line2D([], [], color='0.3', ls='--', lw=0.7, label='Exp. fit'),
        Line2D([], [], color='0.3', ls=':', lw=0.7, label='Model'),
    ]
    ax_08.legend(
        handles=ax_08.get_legend_handles_labels()[0] + style_handles,
        fontsize=5, loc='upper right')

    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_tau_vs_distance(df_metrics, save_path):
    """Fig 5: exponential decay time constant vs distance (excludes 10 cm)."""
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.6))

    for grams, marker in [(0.2, 'o'), (0.8, 's')]:
        sub = df_metrics[
            (df_metrics['grams'] == grams) &
            (df_metrics['distance_cm'] > 10)
        ].sort_values('distance_cm')
        valid = sub['decay_tau_s'].notna()
        ax.plot(sub.loc[valid, 'distance_cm'], sub.loc[valid, 'decay_tau_s'],
                marker=marker, color=LOADING_COLORS[grams],
                markersize=4, linewidth=0.9, label=f'{grams} g')

    ax.set_xlabel('True distance (cm)')
    ax.set_ylabel('$\\tau$ (s)')
    ax.set_title('Exponential decay time constant')
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_residual_analysis(residual_data, save_path):
    """Fig 6: 2x1 — residual (experiment - model) vs time."""
    setup_ieee_style()
    fig, (ax_02, ax_08) = plt.subplots(
        2, 1, figsize=(IEEE_SINGLE_COL, 4.8))

    for idx, (ax, grams, title) in enumerate(
        [(ax_02, 0.2, '0.2 g'), (ax_08, 0.8, '0.8 g')]
    ):
        for d_cm, t_min, residual in residual_data[grams]:
            ax.plot(t_min, residual, color=DISTANCE_COLORS[d_cm],
                    linewidth=0.9, label=f'{d_cm} cm')
        ax.axhline(0, color='0.4', ls='--', lw=0.6)
        if idx == 1:
            ax.set_xlabel('Time (min)')
        ax.set_ylabel('Residual (cm)')
        ax.set_title(f'Experiment $-$ Model ({title})')
        ax.legend(title='True dist.', fontsize=6, title_fontsize=7)

    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_anova_table(anova_table, save_path):
    """Fig 7: rendered ANOVA results table."""
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 1.5))
    ax.axis('off')

    col_labels = ['Source', 'SS', 'DF', 'F', 'p-value']
    table_data = []
    for source in anova_table.index:
        row = anova_table.loc[source]
        name = (source
                .replace('C(grams)', 'Dust loading')
                .replace('C(distance_cm)', 'Distance')
                .replace(':', ' x '))

        f_val = row.get('F', np.nan)
        p_val = row.get('PR(>F)', np.nan)
        f_str = f'{f_val:.2f}' if np.isfinite(f_val) else '--'
        sig = ' *' if np.isfinite(p_val) and p_val < 0.05 else ''
        p_str = f'{p_val:.4f}{sig}' if np.isfinite(p_val) else '--'

        table_data.append([
            name,
            f'{row["sum_sq"]:.1f}',
            f'{row["df"]:.0f}',
            f_str,
            p_str,
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#DDDDDD')
        table[0, j].set_text_props(weight='bold')

    for i, source in enumerate(anova_table.index):
        p_val = anova_table.loc[source].get('PR(>F)', np.nan)
        if np.isfinite(p_val) and p_val < 0.05:
            for j in range(len(col_labels)):
                table[i + 1, j].set_text_props(weight='bold')

    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    print(f'Saved: {save_path}')


def generate_plots():
    output_dir = os.path.join(os.path.dirname(__file__), 'ir_analysis_plots')
    os.makedirs(output_dir, exist_ok=True)

    kern = 32

    control_data = {}
    for grams in DUST_LOADINGS_G:
        true_dists, mean_dists, std_dists = [], [], []
        for d_cm in DISTANCES_CM:
            df = load_trial1_data(grams, d_cm)
            control = df[df['phase_type'] == 'control']['distance_cm_smooth']
            true_dists.append(d_cm)
            mean_dists.append(control.mean())
            std_dists.append(control.std())
        control_data[grams] = (true_dists, mean_dists, std_dists)
    plot_measured_vs_actual(control_data, os.path.join(
        output_dir, 'fig1_control_phase_baseline_accuracy.png'))

    error_comp_data = {}
    for grams in DUST_LOADINGS_G:
        error_comp_data[grams] = []
        for d_cm in DISTANCES_CM:
            trial_errors = []
            for trial in TRIALS:
                df = load_trial_data(grams, d_cm, trial)
                sub = df[df['phase_type'] == 'measurement']
                t_blk, d_blk = block_average(
                    phase_time_seconds(sub),
                    sub['distance_cm_smooth'].values,
                )
                t_min = t_blk / 60.0
                err = pd.Series(d_blk - d_cm).rolling(
                    kern, min_periods=1, center=True).mean().values
                trial_errors.append((t_min, err))

            sim = simulate_experiment(distance_cm=d_cm, mass_g=grams)
            model_t_min = sim['t_s'] / 60.0
            error_comp_data[grams].append((
                d_cm, trial_errors,
                model_t_min, sim['distance_est_cm'] - d_cm,
            ))

    plot_error_comparison_grid(error_comp_data, os.path.join(
        output_dir, 'fig2_experimental_vs_model_error.png'))

    plot_error_comparison_grid_pct(error_comp_data, os.path.join(
        output_dir, 'fig2_v2_experimental_vs_model_pct_error.png'))

    plot_error_comparison_overlay_pct(error_comp_data, os.path.join(
        output_dir, 'fig2_v3_overlay_pct_error.png'))

    plot_error_comparison_overlay_pct_wide(error_comp_data, os.path.join(
        output_dir, 'fig2_v4_overlay_pct_error_wide.png'))

    plot_error_comparison_overlay_pct_v5(error_comp_data, os.path.join(
        output_dir, 'fig2_v5_overlay_pct_error_mean_std.png'))

    print('\nComputing metrics ...')
    df_metrics = compute_all_metrics()
    csv_path = os.path.join(output_dir, 'metrics.csv')
    df_metrics.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')
    save_summary_table(df_metrics, os.path.join(
        output_dir, 'summary_table.csv'))

    power_fits = {}
    for grams in DUST_LOADINGS_G:
        sub = df_metrics[df_metrics['grams'] == grams].sort_values(
            'distance_cm')
        for col in ['early_mae_60s', 'late_mae_120s']:
            power_fits[(grams, col)] = fit_error_vs_distance(
                sub['distance_cm'].values, sub[col].values)
    plot_distance_scaling(df_metrics, os.path.join(
        output_dir, 'fig3_mae_distance_scaling_early_vs_late.png'),
        power_fits=power_fits)

    decay_data = {}
    residual_data = {}
    for grams in DUST_LOADINGS_G:
        decay_data[grams] = []
        residual_data[grams] = []
        for d_cm in DISTANCES_CM:
            t_block, dist_mean, _ = multi_trial_block_average(grams, d_cm)
            error = dist_mean - d_cm
            t_min = t_block / 60.0

            decay = fit_exponential_decay(t_block, error)
            if not np.isnan(decay['tau_s']):
                fit_curve = (decay['A'] * np.exp(-t_block / decay['tau_s'])
                             + decay['C'])
            else:
                fit_curve = None
            sim = simulate_experiment(distance_cm=d_cm, mass_g=grams)
            model_err = np.interp(t_block, sim['t_s'],
                                  sim['distance_est_cm'] - d_cm)

            decay_data[grams].append((
                d_cm, t_min, error, fit_curve, decay['tau_s'], model_err))
            residual_data[grams].append((d_cm, t_min, error - model_err))

    plot_exponential_decay_fits(decay_data, os.path.join(
        output_dir, 'fig4_exponential_decay_fits.png'))

    plot_tau_vs_distance(df_metrics, os.path.join(
        output_dir, 'fig5_tau_vs_distance.png'))

    plot_residual_analysis(residual_data, os.path.join(
        output_dir, 'fig6_residual_analysis.png'))

    print('\nComputing ANOVA ...')
    anova_table = compute_anova()
    anova_path = os.path.join(output_dir, 'anova_results.csv')
    anova_table.to_csv(anova_path)
    print(f'Saved: {anova_path}')
    plot_anova_table(anova_table, os.path.join(
        output_dir, 'fig7_anova_table.png'))

    save_paper_summary(df_metrics, anova_table,
                       os.path.join(output_dir, 'paper_summary.csv'))

    print(f'\nAll figures saved to {output_dir}/')


if __name__ == '__main__':
    generate_plots()
