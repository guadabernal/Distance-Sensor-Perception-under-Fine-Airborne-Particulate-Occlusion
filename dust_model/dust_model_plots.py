"""Generate dust settling model figures and concentration table."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import (
    setup_ieee_style, save_fig, IEEE_SINGLE_COL, IEEE_DOUBLE_COL,
    BIN_COLORS, BIN_MARKERS, LOADING_COLORS,
)

from dust_model import run_dust_model, T_TOTAL_S, SETTLING_TIME_S


def _panel_label(ax, label):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ylab_bbox = ax.yaxis.get_label().get_window_extent(renderer)
    x_left = ax.transAxes.inverted().transform((ylab_bbox.x0, 0))[0]
    ax.text(x_left - 0.035, 1.0, label, transform=ax.transAxes,
            ha='left', va='top', fontsize=9, fontweight='bold')


def plot_normalized(df_ts, df_bins, t_total_s, vline_time_s, save_path,
                    t_axis_end_min=None, fig_height=2.8, panel_label=None):
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL * 0.6, fig_height))

    t_end_min = t_total_s / 60.0
    if t_axis_end_min is not None:
        t_end_min = min(t_end_min, t_axis_end_min)
    vline_min = vline_time_s / 60.0
    ax.axvspan(0, vline_min, alpha=0.12, color='gray', label='Settling')
    ax.axvspan(vline_min, t_end_min, alpha=0.12, color='#88CCEE',
               label='Measurement')

    c0_tot = df_ts['C_total_mg_m3'].iloc[0]
    ax.plot(df_ts['time_min'], df_ts['C_total_mg_m3'] / c0_tot * 100,
            color='black', linewidth=1.2, label='Total')

    marker_every = max(1, len(df_ts) // 15)
    for i in range(len(df_bins)):
        col = f'C_{i + 1}_mg_m3'
        c0 = df_ts[col].iloc[0]
        norm = df_ts[col] / c0 * 100 if c0 > 0 else df_ts[col] * 0
        row = df_bins.iloc[i]
        ax.plot(df_ts['time_min'], norm,
                color=BIN_COLORS[i % len(BIN_COLORS)],
                marker=BIN_MARKERS[i % len(BIN_MARKERS)],
                markevery=marker_every, markersize=3, linewidth=0.8,
                label=f'{row["d_low_um"]:.1f}-{row["d_high_um"]:.1f} $\\mu$m')

    ax.set_xlim(0, t_end_min)
    ax.set_ylim(0, 105)
    ax.set_xticks(np.arange(0, t_end_min + 1.25, 2.5))
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('$C/C_0$ (%)')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=6, frameon=True, loc='center left',
              bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout(rect=[0, 0, 0.98, 1])
    if panel_label:
        _panel_label(ax, panel_label)
    save_fig(fig, save_path)
    plt.close(fig)


def plot_cair_total(df_ts_by_loading, settling_time_s, save_path,
                    t_axis_end_min=17.5, fig_height=2.1, panel_label=None):
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL * 0.6, fig_height))

    settle_min = settling_time_s / 60.0
    ax.axvspan(0, settle_min, alpha=0.12, color='gray', label='Settling')
    ax.axvspan(settle_min, t_axis_end_min, alpha=0.12, color='#88CCEE',
               label='Measurement')

    y_top = 0.0
    for grams, df_ts in df_ts_by_loading.items():
        t = df_ts['time_min'].values
        c_g = df_ts['C_total_mg_m3'].values / 1000.0
        mask = t <= t_axis_end_min + 1e-9
        color = LOADING_COLORS[grams]
        ax.plot(t[mask], c_g[mask], color=color, linewidth=1.2,
                label=f'{grams} g injection', zorder=3)
        y_top = max(y_top, c_g[mask].max())

        c_start = np.interp(0.0, t, c_g)
        c_open = np.interp(settle_min, t, c_g)
        c_end = np.interp(t_axis_end_min, t, c_g)
        ax.plot([0.0, settle_min, t_axis_end_min], [c_start, c_open, c_end],
                linestyle='none', marker='o', markersize=3.5, color=color,
                zorder=4, clip_on=False)
        for x, c, ha, offset in ((0.0, c_start, 'left', (5, 3)),
                                 (settle_min, c_open, 'left', (5, 4)),
                                 (t_axis_end_min, c_end, 'right', (-3, 6))):
            ax.annotate(f'{c:.3f}', xy=(x, c), xytext=offset,
                        textcoords='offset points', ha=ha,
                        fontsize=6.5, color=color, zorder=5)

    ax.set_xlim(0, t_axis_end_min)
    ax.set_ylim(0, y_top * 1.10)
    ax.set_xticks(np.arange(0, t_axis_end_min + 0.1, 2.5))
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('$C_{\\mathrm{air}}$ (g m$^{-3}$)')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=6, frameon=True, loc='center left',
              bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout(rect=[0, 0, 0.98, 1])
    if panel_label:
        _panel_label(ax, panel_label)
    save_fig(fig, save_path)
    plt.close(fig)


def generate_plots():
    output_dir = os.path.join(os.path.dirname(__file__), 'dust_model_plots')
    os.makedirs(output_dir, exist_ok=True)

    df_bins_02, df_ts_02, C0_02 = run_dust_model(0.2)
    df_bins_08, df_ts_08, C0_08 = run_dust_model(0.8)

    plot_normalized(df_ts_02, df_bins_02, T_TOTAL_S, SETTLING_TIME_S,
                    os.path.join(output_dir, 'normalized.png'))

    plot_normalized(df_ts_02, df_bins_02, T_TOTAL_S, SETTLING_TIME_S,
                    os.path.join(output_dir, 'normalized_v2.png'),
                    t_axis_end_min=17.5, fig_height=2.1, panel_label='(b)')

    plot_cair_total({0.2: df_ts_02, 0.8: df_ts_08}, SETTLING_TIME_S,
                    os.path.join(output_dir, 'normalized_v3.png'),
                    panel_label='(a)')

    i_settle = int(np.argmin(np.abs(df_ts_02['time_s'] - SETTLING_TIME_S)))
    table_data = []
    for i in range(len(df_bins_02)):
        row = df_bins_02.iloc[i]
        col = f'C_{i + 1}_mg_m3'
        table_data.append({
            'Bin': f'{row["d_low_um"]:.1f}-{row["d_high_um"]:.0f} um',
            '0.2g t=0 (mg/m^3)': round(df_ts_02[col].iloc[0], 1),
            '0.2g t=150s (mg/m^3)': round(df_ts_02[col].iloc[i_settle], 1),
            '0.8g t=0 (mg/m^3)': round(df_ts_08[col].iloc[0], 1),
            '0.8g t=150s (mg/m^3)': round(df_ts_08[col].iloc[i_settle], 1),
        })
    table_data.append({
        'Bin': 'Total',
        '0.2g t=0 (mg/m^3)': round(C0_02, 1),
        '0.2g t=150s (mg/m^3)': round(df_ts_02['C_total_mg_m3'].iloc[i_settle], 1),
        '0.8g t=0 (mg/m^3)': round(C0_08, 1),
        '0.8g t=150s (mg/m^3)': round(df_ts_08['C_total_mg_m3'].iloc[i_settle], 1),
    })

    table_path = os.path.join(output_dir, 'initial_and_measurement_start_concentrations.csv')
    pd.DataFrame(table_data).to_csv(table_path, index=False)
    print(f"Saved: {table_path}")


if __name__ == '__main__':
    generate_plots()
