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
    setup_ieee_style, save_fig, IEEE_DOUBLE_COL, BIN_COLORS, BIN_MARKERS,
)

from dust_model import run_dust_model, T_TOTAL_S, SETTLING_TIME_S


def plot_normalized(df_ts, df_bins, t_total_s, vline_time_s, save_path):
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL * 0.6, 2.8))

    t_end_min = t_total_s / 60.0
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
    save_fig(fig, save_path)
    plt.close(fig)


def generate_plots():
    output_dir = os.path.join(os.path.dirname(__file__), 'dust_model_plots')
    os.makedirs(output_dir, exist_ok=True)

    df_bins_02, df_ts_02, C0_02 = run_dust_model(0.2)
    df_bins_08, df_ts_08, C0_08 = run_dust_model(0.8)

    plot_normalized(df_ts_02, df_bins_02, T_TOTAL_S, SETTLING_TIME_S,
                    os.path.join(output_dir, 'normalized.png'))

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
