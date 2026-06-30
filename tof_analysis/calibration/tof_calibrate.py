"""Fit VL53L5CX ToF observation parameters: false-target evidence + wall-locked bias closure."""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from scipy.optimize import differential_evolution

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))

from tof_model import (  # noqa: E402
    ToFModelParams, DEFAULT_PARAMS, replace,
    build_psd, build_dust_data, evidence_series,
    concentration_scale_for_mass, SIM_T_MAX_MIN,
)
from exp_targets import EXP_TARGETS_02, EXP_TARGETS_08  # noqa: E402

DISTANCES_CM = [10, 20, 30, 40, 50, 60]
DISTANCES_M = [d / 100.0 for d in DISTANCES_CM]
TIME_BIN_WIDTH_S = 30.0
FL_THRESHOLD_FRAC = 0.70

N_FRAMES_CAL = 64
T_GRID_S = np.concatenate([
    np.linspace(0.0, 5.0 * 60.0, 61),
    np.linspace(5.5 * 60.0, SIM_T_MAX_MIN * 60.0, 20),
])

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

W_FL = 16.0
W_SNAP = 5.0
W_BIAS = 4.00
SCALE_RATIO_REG_WEIGHT = 0.10

_BASE_DUST = None
_BASE_PSD = None


def _snap_time_from_fl(time_s, f_fl, threshold=0.05, n_confirm=3):
    ok = np.asarray(f_fl) < threshold
    if len(ok) == 0:
        return np.nan
    if np.all(ok):
        return 0.0
    for i in range(len(ok) - 1, -1, -1):
        if not ok[i]:
            c = i + 1
            if c + n_confirm <= len(ok):
                return float(time_s[c])
            return np.nan
    return np.nan


def _metrics_from_evidence(ev, t_grid_s, dist_cm):
    p_wall = np.clip(np.asarray(ev['p_wall'], dtype=float), 0.0, 1.0)
    f_fl_t = 1.0 - p_wall
    edges = np.arange(0.0, t_grid_s[-1] + TIME_BIN_WIDTH_S, TIME_BIN_WIDTH_S)
    centers = (edges[:-1] + edges[1:]) / 2.0
    fl_fracs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (t_grid_s >= lo) & (t_grid_s < hi)
        fl_fracs.append(float(np.nanmean(f_fl_t[mask])) if np.any(mask) else np.nan)
    fl_fracs = np.asarray(fl_fracs)
    first = centers <= 30.0
    fl_initial = float(np.nanmean(fl_fracs[first])) if np.any(first) else np.nan
    snap_s = _snap_time_from_fl(centers, fl_fracs)
    snap_min = snap_s / 60.0 if np.isfinite(snap_s) else 15.0

    early = t_grid_s <= 180.0
    wall_bias_cm = np.asarray(ev['wall_bias_m'], dtype=float) * 100.0
    bias_mask = early & (p_wall > 0.05) & np.isfinite(wall_bias_cm)
    peak_neg_bias = float(np.nanmin(wall_bias_cm[bias_mask])) if np.any(bias_mask) else 0.0
    return {'fl_initial': fl_initial, 'snap_time_min': snap_min,
            'peak_neg_bias_cm': peak_neg_bias}


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
    _, base_dust = _get_base()
    base = base_dust[float(mass_g)]
    scale = concentration_scale_for_mass(mass_g, params)
    out = dict(base)
    for k in ['C_bins_mg_m3', 'C_total_mg_m3', 'alpha_ext_1_per_m',
              'alpha_sca_1_per_m', 'beta_mie_1_per_m_sr',
              'beta_back_1_per_m_sr', 'wall_bias_driver_1_per_m']:
        if k in base:
            out[k] = np.asarray(base[k]) * scale
    out['concentration_scale'] = scale
    out['q_window'] = np.asarray(base.get('q_window', np.zeros_like(base['t_s'])))
    return out


def run_forward_model(params: ToFModelParams, mass_g, n_frames=N_FRAMES_CAL, seed=42):
    _ = n_frames, seed
    psd_cache, _ = _get_base()
    psd = psd_cache[float(mass_g)]
    dust_data = _scaled_dust_data(float(mass_g), params)
    t_min = T_GRID_S / 60.0
    out = {}
    for d_cm, D in zip(DISTANCES_CM, DISTANCES_M):
        ev = evidence_series(D, t_min, mass_g, psd=psd, dust_data=dust_data, params=params)
        out[d_cm] = _metrics_from_evidence(ev, T_GRID_S, d_cm)
    return out


def cost_against_targets(metrics, targets):
    cost = 0.0
    for d, target in targets.get('fl_initial', {}).items():
        if d in metrics:
            cost += W_FL * (metrics[d]['fl_initial'] - target) ** 2
    for d, target in targets.get('snap_time_min', {}).items():
        if d in metrics:
            val = metrics[d]['snap_time_min']
            cost += W_SNAP * ((val - target) / max(target, 0.25)) ** 2
    for d, target in targets.get('peak_neg_bias_cm', {}).items():
        if d in metrics:
            val = metrics[d]['peak_neg_bias_cm']
            scale = max(abs(target), 0.25)
            cost += W_BIAS * ((val - target) / scale) ** 2
    return float(cost)


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
    cost = cost_against_targets(m08, EXP_TARGETS_08) + cost_against_targets(m02, EXP_TARGETS_02)
    log_ratio = np.log10(max(p.concentration_scale_0p2, 1e-30) / max(p.concentration_scale_0p8, 1e-30))
    cost += SCALE_RATIO_REG_WEIGHT * log_ratio ** 2
    return float(cost)


def print_metrics(metrics, title):
    print('\n' + title)
    print('  dist  FL_init  snap(min)  peak_bias(cm)')
    for d in DISTANCES_CM:
        m = metrics[d]
        print(f'  {d:3d}   {m["fl_initial"]:7.3f}   {m["snap_time_min"]:8.2f}'
              f'   {m["peak_neg_bias_cm"]:+10.3f}')


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
    print_metrics(run_forward_model(DEFAULT_PARAMS, 0.8), 'Default model (0.8 g)')
    print_metrics(run_forward_model(DEFAULT_PARAMS, 0.2), 'Default model (0.2 g)')
    print('\nRunning differential evolution for revised joint model...')
    result = differential_evolution(objective_joint, BOUNDS_JOINT, seed=42,
                                    maxiter=maxiter, popsize=4, tol=1e-3,
                                    polish=False, workers=1)
    p = params_from_vector_joint(result.x)
    m08 = run_forward_model(p, 0.8)
    m02 = run_forward_model(p, 0.2)
    print_metrics(m08, 'Fitted model (0.8 g)')
    print_metrics(m02, 'Fitted model (0.2 g)')
    print('\nFitted parameter block:')
    for name in ALL_PARAM_NAMES:
        print(f'  {name} = {getattr(p, name):.9g}')
    print(f'  objective = {result.fun:.6g}')
    out_path = os.path.join(OUT_DIR, 'calibrated_params_revised_joint.txt')
    write_params_file(p, result.fun, out_path, mode='revised_joint')
    print(f'Saved {out_path}')
    return result


def fit_bias_only():
    """Second-stage fit of the wall-bias closure with lock params fixed."""
    from scipy.optimize import least_squares

    target_rows = []
    for mass_g, targets in [(0.2, EXP_TARGETS_02), (0.8, EXP_TARGETS_08)]:
        for d_cm, target_cm in targets['peak_neg_bias_cm'].items():
            if d_cm > 40:
                continue
            target_rows.append((mass_g, d_cm, target_cm))

    def params_from_x(x):
        log_gain, size_power, driver_power, center, width = x
        return replace(DEFAULT_PARAMS,
                       wall_bias_centroid_gain=10.0 ** log_gain,
                       wall_bias_size_power=float(size_power),
                       wall_bias_driver_power=float(driver_power),
                       wall_bias_distance_center_m=float(center),
                       wall_bias_distance_width_m=float(width),
                       bias_scale=0.0)

    def residual(x):
        p = params_from_x(x)
        out = []
        for mass_g, d_cm, target_cm in target_rows:
            dust = build_dust_data(mass_g, t_meas_s=np.array([0.0]), params=p)
            ev = evidence_series(d_cm / 100.0, np.array([0.0]), mass_g,
                                 dust_data=dust, params=p)
            pred_cm = float(ev['wall_bias_m'][0]) * 100.0
            scale = max(abs(target_cm), 0.25)
            w = 2.0 if (mass_g == 0.8 and d_cm in (20, 30, 40)) else 1.0
            out.append(np.sqrt(w) * (pred_cm - target_cm) / scale)
        return np.asarray(out)

    x0 = np.array([
        np.log10(DEFAULT_PARAMS.wall_bias_centroid_gain),
        DEFAULT_PARAMS.wall_bias_size_power,
        DEFAULT_PARAMS.wall_bias_driver_power,
        DEFAULT_PARAMS.wall_bias_distance_center_m,
        DEFAULT_PARAMS.wall_bias_distance_width_m,
    ])
    lower = [-4.0, 0.0, 0.5, 0.12, 0.03]
    upper = [1.0, 8.0, 3.0, 0.34, 0.20]
    res = least_squares(residual, x0, bounds=(lower, upper))
    p = params_from_x(res.x)
    print('\nBias-only fitted parameter block:')
    for name in ['wall_bias_centroid_gain', 'wall_bias_size_power',
                 'wall_bias_driver_power', 'wall_bias_distance_center_m',
                 'wall_bias_distance_width_m', 'bias_scale']:
        print(f'  {name} = {getattr(p, name):.10g}')
    print(f'  normalized RMS residual = {np.sqrt(np.mean(res.fun**2)):.4f}')
    return p, res

def main():
    ap = argparse.ArgumentParser(description='Fit revised ToF observation model')
    ap.add_argument('--maxiter', type=int, default=80)
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--bias-only', action='store_true')
    args = ap.parse_args()
    if args.bias_only:
        fit_bias_only()
    elif args.eval_only:
        print_metrics(run_forward_model(DEFAULT_PARAMS, 0.8), 'Default model (0.8 g)')
        print_metrics(run_forward_model(DEFAULT_PARAMS, 0.2), 'Default model (0.2 g)')
    else:
        fit_joint(maxiter=args.maxiter)


if __name__ == '__main__':
    main()
