"""Generate IR dust-observation model figures (alpha/transmittance, distance bias, T vs distance)."""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import (
    setup_ieee_style, save_fig,
    IEEE_SINGLE_COL, IEEE_DOUBLE_COL,
    TOL_MUTED, LOADING_COLORS, DISTANCE_COLORS,
)

from ir_model import simulate_experiment


def plot_beta_and_transmittance(results_02, results_08, save_path):
    """Extinction coefficient and round-trip transmittance vs time at d=50 cm."""
    setup_ieee_style()
    fig, (ax_b, ax_t) = plt.subplots(
        1, 2, figsize=(IEEE_DOUBLE_COL, 2.8))

    t_min_02 = results_02['t_s'] / 60.0
    t_min_08 = results_08['t_s'] / 60.0
    c02, c08 = LOADING_COLORS[0.2], LOADING_COLORS[0.8]

    ax_b.plot(t_min_02, results_02['alpha_ext_1_per_m'], color=c02, label='0.2 g')
    ax_b.plot(t_min_08, results_08['alpha_ext_1_per_m'], color=c08, label='0.8 g')
    ax_b.axvline(0, color='0.4', ls=':', lw=0.7, label='Measurement start')
    ax_b.set_xlabel('Time (min)')
    ax_b.set_ylabel(r'$\alpha_{\rm ext}$ (1/m)')
    ax_b.set_title('Beer-Lambert extinction (d = 50 cm)')
    ax_b.legend()

    ax_t.plot(t_min_02, results_02['T_roundtrip'], color=c02, label='0.2 g')
    ax_t.plot(t_min_08, results_08['T_roundtrip'], color=c08, label='0.8 g')
    ax_t.axvline(0, color='0.4', ls=':', lw=0.7, label='Measurement start')
    ax_t.set_xlabel('Time (min)')
    ax_t.set_ylabel('Round-trip transmittance')
    ax_t.set_title('Transmittance (d = 50 cm)')
    ax_t.legend()

    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_distance_bias(bias_data, save_path):
    """Distance error vs time for multiple true distances."""
    setup_ieee_style()
    fig, (ax_lo, ax_hi) = plt.subplots(
        1, 2, figsize=(IEEE_DOUBLE_COL, 2.8))

    for ax, mass, title in [(ax_lo, 0.2, '0.2 g'), (ax_hi, 0.8, '0.8 g')]:
        for d_cm, t_min, error_cm in bias_data[mass]:
            ax.plot(t_min, error_cm, color=DISTANCE_COLORS[d_cm],
                    label=f'{d_cm} cm')
        ax.axhline(0, color='0.4', ls='--', lw=0.6)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Distance error (cm)')
        ax.set_title(f'Distance error ({title})')
        ax.legend(title='True dist.', fontsize=6, title_fontsize=7)

    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_transmittance_vs_distance(transmittance_data, save_path):
    """Worst-case transmittance at measurement start vs true distance."""
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.6))

    for mass, color, label in [(0.2, LOADING_COLORS[0.2], '0.2 g'),
                                (0.8, LOADING_COLORS[0.8], '0.8 g')]:
        distances_cm, T_at_t0 = transmittance_data[mass]
        ax.plot(distances_cm, T_at_t0, color=color, label=label)

    ax.set_xlabel('True distance (cm)')
    ax.set_ylabel('Round-trip transmittance at measurement start')
    ax.set_title('Transmittance at measurement start vs distance')
    ax.legend()

    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    print(f'Saved: {save_path}')


def generate_plots():
    output_dir = os.path.join(os.path.dirname(__file__), 'ir_model_plots')
    os.makedirs(output_dir, exist_ok=True)

    res_02 = simulate_experiment(distance_cm=50, mass_g=0.2)
    res_08 = simulate_experiment(distance_cm=50, mass_g=0.8)
    plot_beta_and_transmittance(
        res_02, res_08,
        os.path.join(output_dir, 'beta_and_transmittance.png'))

    true_distances_cm = [10, 20, 30, 40, 50, 60]
    bias_data = {}
    for mass in [0.2, 0.8]:
        bias_data[mass] = []
        for d_cm in true_distances_cm:
            res = simulate_experiment(distance_cm=d_cm, mass_g=mass)
            t_min = res['t_s'] / 60.0
            error_cm = res['distance_est_cm'] - d_cm
            bias_data[mass].append((d_cm, t_min, error_cm))
    plot_distance_bias(
        bias_data, os.path.join(output_dir, 'distance_bias.png'))

    distances_cm = np.arange(5, 65, 1)
    transmittance_data = {}
    for mass in [0.2, 0.8]:
        T_at_t0 = []
        for d_cm in distances_cm:
            res = simulate_experiment(distance_cm=d_cm, mass_g=mass)
            T_at_t0.append(res['T_roundtrip'][0])
        transmittance_data[mass] = (distances_cm, T_at_t0)
    plot_transmittance_vs_distance(
        transmittance_data,
        os.path.join(output_dir, 'transmittance_vs_distance.png'))


if __name__ == '__main__':
    generate_plots()
