"""Fit VL53L5CX ToF observation parameters to the experimental false-lock and wall-locked bias time series."""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))

from tof_model import (  # noqa: E402
    ToFModelParams, DEFAULT_PARAMS, replace,
    build_psd, build_dust_data, evidence_series, size_weighted_bias_driver,
    concentration_scale_for_mass, SIM_T_MAX_MIN,
)

DISTANCES_CM = [10, 20, 30, 40, 50, 60]
DISTANCES_M = [d / 100.0 for d in DISTANCES_CM]
TIME_BIN_WIDTH_S = 30.0

T_GRID_S = np.concatenate([
    np.linspace(0.0, 5.0 * 60.0, 61),
    np.linspace(5.5 * 60.0, SIM_T_MAX_MIN * 60.0, 20),
])

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_TIMESERIES_PATH = os.path.join(os.path.dirname(_THIS_DIR), 'metrics', 'timeseries_metrics.csv')

W_FL = 200.0
W_BIAS = 4.00
SCALE_RATIO_REG_WEIGHT = 0.10

TURB_NODES, TURB_WEIGHTS = np.polynomial.hermite_e.hermegauss(7)
TURB_WEIGHTS = TURB_WEIGHTS / TURB_WEIGHTS.sum()

_BASE_DUST = None
_BASE_PSD = None
_EXP_CURVES = None


def _curves_from_evidence(ev, t_grid_s):
    f_fl_t = 1.0 - np.clip(np.asarray(ev['p_wall'], dtype=float), 0.0, 1.0)
    wall_bias_cm = np.asarray(ev['wall_bias_m'], dtype=float) * 100.0
    bins = np.floor(t_grid_s / TIME_BIN_WIDTH_S).astype(int)
    n_bins = int(round(SIM_T_MAX_MIN * 60.0 / TIME_BIN_WIDTH_S))
    f_fl = np.array([f_fl_t[bins == i].mean() for i in range(n_bins)])
    bias = np.array([wall_bias_cm[bins == i].mean() for i in range(n_bins)])
    return {'f_fl': f_fl, 'bias_cm': bias}


def _get_exp_curves():
    global _EXP_CURVES
    if _EXP_CURVES is None:
        ts = pd.read_csv(EXP_TIMESERIES_PATH)
        _EXP_CURVES = {}
        for (g, d), sub in ts.groupby(['loading_g', 'dist_cm']):
            sub = sub.sort_values('time_bin_center_s')
            _EXP_CURVES[(float(g), int(d))] = {'f_fl': sub['f_fl_exp'].values,
                                               'bias_cm': sub['bias_wall_mean_exp'].values}
    return _EXP_CURVES


def _base_params_for_cache():
    return replace(DEFAULT_PARAMS, concentration_scale_0p2=1.0, concentration_scale_0p8=1.0)


def _get_base():
    global _BASE_DUST, _BASE_PSD
    if _BASE_DUST is None or _BASE_PSD is None:
        p0 = _base_params_for_cache()
        _BASE_DUST = {
            0.2: build_dust_data(0.2, t_meas_s=T_GRID_S, params=p0),
            0.8: build_dust_data(0.8, t_meas_s=T_GRID_S, params=p0),
        }
        _BASE_PSD = {0.2: build_psd(0.2), 0.8: build_psd(0.8)}
    return _BASE_PSD, _BASE_DUST


def _scaled_dust_data(mass_g, params):
    psd_cache, base_dust = _get_base()
    base = base_dust[float(mass_g)]
    scale = concentration_scale_for_mass(mass_g, params)
    out = dict(base)
    for k in ['C_bins_mg_m3', 'C_total_mg_m3', 'alpha_ext_1_per_m',
              'alpha_sca_1_per_m', 'beta_mie_1_per_m_sr',
              'beta_back_1_per_m_sr']:
        if k in base:
            out[k] = np.asarray(base[k]) * scale
    out['wall_bias_driver_1_per_m'] = scale * size_weighted_bias_driver(
        base['alpha_bins_ext_1_per_m'], psd_cache[float(mass_g)]['d_rep_um'], params)
    out['concentration_scale'] = scale
    out['q_window'] = np.asarray(base.get('q_window', np.zeros_like(base['t_s'])))
    return out


def _turbulence_averaged_evidence(D, t_min, mass_g, psd, dust_data, params):
    """Expected p_wall and wall-locked bias over the lognormal alpha/beta turbulence of measure_one_frame."""
    sigma = float(params.sigma_turb)
    p_wall, bias_num = 0.0, 0.0
    for z, w in zip(TURB_NODES, TURB_WEIGHTS):
        mult = np.exp(sigma * z - 0.5 * sigma ** 2)
        dd = dict(dust_data)
        dd['alpha_ext_1_per_m'] = dust_data['alpha_ext_1_per_m'] * mult
        dd['beta_back_1_per_m_sr'] = dust_data['beta_back_1_per_m_sr'] * mult
        ev = evidence_series(D, t_min, mass_g, psd=psd, dust_data=dd, params=params)
        p_wall = p_wall + w * ev['p_wall']
        bias_num = bias_num + w * ev['p_wall'] * ev['wall_bias_m']
    wall_bias = np.divide(bias_num, p_wall, out=np.zeros_like(p_wall), where=p_wall > 0)
    return {'p_wall': p_wall, 'wall_bias_m': wall_bias}


def run_forward_model(params: ToFModelParams, mass_g):
    psd_cache, _ = _get_base()
    psd = psd_cache[float(mass_g)]
    dust_data = _scaled_dust_data(float(mass_g), params)
    t_min = T_GRID_S / 60.0
    out = {}
    for d_cm, D in zip(DISTANCES_CM, DISTANCES_M):
        ev = _turbulence_averaged_evidence(D, t_min, mass_g, psd, dust_data, params)
        out[d_cm] = _curves_from_evidence(ev, T_GRID_S)
    return out


def curve_errors(curves, mass_g):
    exp = _get_exp_curves()
    out = {}
    for d_cm, mod in curves.items():
        e = exp[(float(mass_g), d_cm)]
        n = len(e['f_fl'])
        fl_mse = float(np.nanmean((mod['f_fl'][:n] - e['f_fl']) ** 2))
        bias_scale = max(float(np.nanmax(np.abs(e['bias_cm']))), 0.25)
        bias_nmse = float(np.nanmean(((mod['bias_cm'][:n] - e['bias_cm']) / bias_scale) ** 2))
        out[d_cm] = {'fl_mse': fl_mse, 'bias_nmse': bias_nmse,
                     'bias_rmse_cm': np.sqrt(bias_nmse) * bias_scale}
    return out


def cost_against_experiment(curves, mass_g):
    err = curve_errors(curves, mass_g)
    return float(sum(W_FL * e['fl_mse'] + W_BIAS * e['bias_nmse'] for e in err.values()))


ALL_PARAM_NAMES = [
    'dust_gain',
    'concentration_scale_0p8',
    'concentration_scale_0p2',
    'lock_intercept',
    'lock_gain_evidence',
    'false_target_gain',
    'false_target_alpha_ref_1_per_m',
    'false_target_alpha_power',
    'false_target_range_crit_m',
    'false_target_range_width_m',
    'false_target_decay_tau_ref_s',
    'false_target_decay_distance_power',
    'wall_bias_max_m',
    'wall_bias_centroid_gain',
    'wall_bias_size_power',
    'wall_bias_driver_power',
    'wall_bias_distance_center_m',
    'wall_bias_distance_width_m',
    'estimator_window_sigma_m',
    'bias_scale',
    'sigma_turb',
]

BOUNDS_JOINT = [
    (-5.0, 2.0),
    (-2.0, 1.0),
    (-2.0, 1.0),
    (-12.0, 12.0),
    (0.0, 8.0),
    (-2.0, 6.0),
    (0.05, 4.0),
    (0.5, 8.0),
    (0.25, 0.45),
    (0.005, 0.08),
    (5.0, 180.0),
    (0.0, 5.0),
    (0.003, 0.08),
    (-4.0, 1.0),
    (0.0, 8.0),
    (0.5, 3.0),
    (0.12, 0.34),
    (0.03, 0.20),
    (0.015, 0.20),
    (0.0, 5.0),
    (0.02, 0.80),
]


def params_from_vector_joint(x):
    (log10_gain, log10_scale08, log10_scale02, lock_intercept,
     lock_gain_evidence, log10_false_gain, false_alpha_ref, false_alpha_power,
     false_range_crit, false_range_width, false_tau_ref, false_tau_power,
     wall_bias_max_m, log10_wall_bias_centroid_gain, wall_bias_size_power,
     wall_bias_driver_power, wall_bias_center_m, wall_bias_width_m,
     estimator_sigma_m, bias_scale, sigma_turb) = x
    return replace(
        DEFAULT_PARAMS,
        dust_gain=10.0 ** log10_gain,
        concentration_scale_0p8=10.0 ** log10_scale08,
        concentration_scale_0p2=10.0 ** log10_scale02,
        lock_intercept=lock_intercept,
        lock_gain_evidence=lock_gain_evidence,
        false_target_gain=10.0 ** log10_false_gain,
        false_target_alpha_ref_1_per_m=false_alpha_ref,
        false_target_alpha_power=false_alpha_power,
        false_target_range_crit_m=false_range_crit,
        false_target_range_width_m=false_range_width,
        false_target_decay_tau_ref_s=false_tau_ref,
        false_target_decay_distance_power=false_tau_power,
        wall_bias_max_m=wall_bias_max_m,
        wall_bias_centroid_gain=10.0 ** log10_wall_bias_centroid_gain,
        wall_bias_size_power=wall_bias_size_power,
        wall_bias_driver_power=wall_bias_driver_power,
        wall_bias_distance_center_m=wall_bias_center_m,
        wall_bias_distance_width_m=wall_bias_width_m,
        estimator_window_sigma_m=estimator_sigma_m,
        bias_scale=bias_scale,
        sigma_turb=sigma_turb,
    )


def objective_joint(x):
    p = params_from_vector_joint(x)
    m08 = run_forward_model(p, mass_g=0.8)
    m02 = run_forward_model(p, mass_g=0.2)
    cost = cost_against_experiment(m08, 0.8) + cost_against_experiment(m02, 0.2)
    log_ratio = np.log10(max(p.concentration_scale_0p2, 1e-30) / max(p.concentration_scale_0p8, 1e-30))
    cost += SCALE_RATIO_REG_WEIGHT * log_ratio ** 2
    return float(cost)


def print_metrics(curves, mass_g, title):
    err = curve_errors(curves, mass_g)
    print('\n' + title)
    print('  dist  FL_rmse  bias_rmse(cm)')
    for d in DISTANCES_CM:
        print(f'  {d:3d}   {np.sqrt(err[d]["fl_mse"]):7.3f}   {err[d]["bias_rmse_cm"]:10.3f}')


def write_params_file(p: ToFModelParams, objective_value: float, path: str, mode: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# Revised VL53L5CX ToF observation-model parameters ({mode} fit)\n')
        for name in ALL_PARAM_NAMES:
            f.write(f'{name} = {getattr(p, name):.10g}\n')
        f.write(f'objective = {objective_value:.10g}\n')


def read_params_file(path: str) -> dict:
    out = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            name, val = line.split('=', 1)
            try:
                out[name.strip()] = float(val.strip())
            except ValueError:
                pass
    return out


def params_from_file(path: str) -> ToFModelParams:
    raw = read_params_file(path)
    kwargs = {name: raw[name] for name in ALL_PARAM_NAMES if name in raw}
    return replace(DEFAULT_PARAMS, **kwargs)


def fit_joint(maxiter=80):
    print('Evaluating revised default model...')
    print_metrics(run_forward_model(DEFAULT_PARAMS, 0.8), 0.8, 'Default model (0.8 g)')
    print_metrics(run_forward_model(DEFAULT_PARAMS, 0.2), 0.2, 'Default model (0.2 g)')
    print('\nRunning differential evolution for revised joint model...')
    result = differential_evolution(objective_joint, BOUNDS_JOINT, seed=42,
                                    maxiter=maxiter, popsize=4, tol=1e-3,
                                    polish=False, workers=1)
    p = params_from_vector_joint(result.x)
    m08 = run_forward_model(p, 0.8)
    m02 = run_forward_model(p, 0.2)
    print_metrics(m08, 0.8, 'Fitted model (0.8 g)')
    print_metrics(m02, 0.2, 'Fitted model (0.2 g)')
    print('\nFitted parameter block:')
    for name in ALL_PARAM_NAMES:
        print(f'  {name} = {getattr(p, name):.9g}')
    print(f'  objective = {result.fun:.6g}')
    out_path = os.path.join(OUT_DIR, 'calibrated_params_revised_joint.txt')
    write_params_file(p, result.fun, out_path, mode='revised_joint')
    print(f'Saved {out_path}')
    return result


def main():
    ap = argparse.ArgumentParser(description='Fit revised ToF observation model')
    ap.add_argument('--maxiter', type=int, default=80)
    ap.add_argument('--eval-only', action='store_true')
    args = ap.parse_args()
    if args.eval_only:
        print_metrics(run_forward_model(DEFAULT_PARAMS, 0.8), 0.8, 'Default model (0.8 g)')
        print_metrics(run_forward_model(DEFAULT_PARAMS, 0.2), 0.2, 'Default model (0.2 g)')
    else:
        fit_joint(maxiter=args.maxiter)


if __name__ == '__main__':
    main()
