"""Generate figures for the corrected VL53L5CX ToF dust model."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from plot_style import setup_ieee_style, save_fig, IEEE_DOUBLE_COL, IEEE_SINGLE_COL, LOADING_COLORS, DISTANCE_COLORS, PANEL_YLIMS
except Exception:
    IEEE_DOUBLE_COL, IEEE_SINGLE_COL = 7.16, 3.5
    LOADING_COLORS = {0.2: 'tab:blue', 0.8: 'tab:red'}
    DISTANCE_COLORS = {10:'C0',20:'C1',30:'C2',40:'C3',50:'C4',60:'C5'}
    PANEL_YLIMS = {10:(0,20),20:(0,30),30:(0,40),40:(0,50),50:(0,60),60:(0,70)}
    def setup_ieee_style(): pass
    def save_fig(fig, path): fig.savefig(path, dpi=300, bbox_inches='tight')

from tof_model import (build_dust_data, time_series, noiseless_series,
                       compute_evidence_series,
                       wall_locked_distance, compute_snap_threshold,
                       SIM_T_MAX_MIN, SIM_N_FRAMES, SETTLING_TIME_S,
                       DEFAULT_PARAMS)

LOADINGS_G = [0.2, 0.8]
DISTANCES_CM = [10, 20, 30, 40, 50, 60]
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics')


def plot_distance_vs_time(mass_g=0.8):
    setup_ieee_style(); os.makedirs(PLOT_DIR, exist_ok=True)
    t_min = np.unique(np.concatenate([np.linspace(0, 10, 160), np.linspace(10, SIM_T_MAX_MIN, 30)]))
    dust = build_dust_data(mass_g)
    fig, axes = plt.subplots(3, 2, figsize=(IEEE_DOUBLE_COL, 6.0)); axes = axes.ravel()
    for ax, d_cm in zip(axes, DISTANCES_CM):
        D = d_cm/100.0
        data = time_series(D, t_min, mass_g, n_frames=SIM_N_FRAMES, dust_data=dust)
        det = noiseless_series(D, t_min, mass_g, dust_data=dust)
        flat_t = np.repeat(t_min, SIM_N_FRAMES); flat_d = data.ravel()*100
        valid = np.isfinite(flat_d); fl = valid & (flat_d < 0.70*d_cm); wall = valid & ~fl
        ax.scatter(flat_t[wall], flat_d[wall], s=0.6, alpha=0.30, color='tab:blue', linewidths=0, rasterized=True)
        ax.scatter(flat_t[fl], flat_d[fl], s=0.8, alpha=0.45, color='tab:red', linewidths=0, rasterized=True)
        ax.plot(t_min, det*100, color='0.15', ls='--', lw=0.8)
        ax.axhline(d_cm, color='tab:green', lw=0.8)
        st, _, _ = compute_snap_threshold(D, mass_g, dust_data=dust)
        if st is not None: ax.axvline(st, color='tab:orange', ls=':', lw=0.8)
        ax.set_title(f'{d_cm} cm'); ax.set_xlim(-0.2, SIM_T_MAX_MIN+0.2)
        ax.set_ylim(*PANEL_YLIMS.get(d_cm, (0, max(70, d_cm+10))))
        ax.set_xlabel('Time since first sample (min)'); ax.set_ylabel('Distance (cm)')
        ax.grid(True, ls='--', lw=0.4, alpha=0.4)
    fig.suptitle(f'Corrected VL53L5CX model, {mass_g} g; measurement starts {SETTLING_TIME_S:.0f} s post-injection\n'
                 f'alpha(t=0)={dust["alpha_ext_1_per_m"][0]:.3g} 1/m, K_dust={DEFAULT_PARAMS.dust_gain:.3g}, '
                 f'K_false={DEFAULT_PARAMS.false_target_gain:.3g}', fontsize=9)
    fig.tight_layout(rect=[0,0,1,0.93])
    path = os.path.join(PLOT_DIR, f'fig_m1_distance_vs_time_{mass_g}g.png')
    save_fig(fig, path); plt.close(fig); print(f'Saved {path}')


def generate_model_metrics():
    rows=[]
    for g in LOADINGS_G:
        dust=build_dust_data(g)
        for d_cm in DISTANCES_CM:
            D=d_cm/100.0; st,a,_=compute_snap_threshold(D,g,dust_data=dust)
            ev=compute_evidence_series(D,dust,mass_g=g)
            d_wall=wall_locked_distance(D,dust['alpha_ext_1_per_m'],dust['beta_back_1_per_m_sr'],q_window=dust['q_window'])*100
            rows.append({'loading_g':g,'dist_cm':d_cm,'alpha_start_1_per_m':dust['alpha_ext_1_per_m'][0],
                         'beta_start_1_per_m_sr':dust['beta_back_1_per_m_sr'][0], 'p_wall_start':ev['p_wall'][0],
                         'snap_time_model_min':st, 'wall_bias_start_cm':d_wall[0]-d_cm, 'wall_bias_end_cm':d_wall[-1]-d_cm})
    df=pd.DataFrame(rows); os.makedirs(METRICS_DIR, exist_ok=True); path=os.path.join(METRICS_DIR,'model_metrics.csv')
    df.to_csv(path,index=False,float_format='%.6g'); print(f'Saved {path}'); return df


def generate_plots():
    for g in LOADINGS_G:
        plot_distance_vs_time(g)
    generate_model_metrics()

if __name__ == '__main__':
    generate_plots()
