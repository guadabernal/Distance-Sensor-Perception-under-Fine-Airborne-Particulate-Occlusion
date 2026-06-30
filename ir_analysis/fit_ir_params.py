"""Fit IR dust-response parameters (eta, per-loading concentration scale, x_reg, x_overlap) via differential evolution."""

from __future__ import annotations

import os
import sys
import numpy as np
from scipy.optimize import differential_evolution

_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, _THIS_DIR)

from ir_model import (  # noqa: E402
    dust_biased_distance,
    compute_measurement_optics,
    SETTLING_TIME_S,
    MEASUREMENT_DURATION_S,
    DT_S,
    ETA_BACKSCATTER,
    X_REG_M,
    X_OVERLAP_M,
)
from ir_analysis import (  # noqa: E402
    multi_trial_block_average,
    DISTANCES_CM,
    DUST_LOADINGS_G,
)


FIT_GRAMS = [0.2, 0.8]
FIT_DISTANCES = [30, 40, 50]


def _precompute_optics():
    """Precompute base alpha/beta at unit concentration scale."""
    print('Precomputing corrected alpha/beta arrays ...')
    t_meas = np.arange(0.0, MEASUREMENT_DURATION_S + DT_S, DT_S)
    cache = {}
    for grams in DUST_LOADINGS_G:
        optics = compute_measurement_optics(
            grams, t_meas, settling_time_s=SETTLING_TIME_S,
            concentration_scale=1.0)
        cache[grams] = optics
    return cache


def _load_experiment():
    print('Loading experimental data ...')
    exp_data = {}
    for grams in DUST_LOADINGS_G:
        for d_cm in DISTANCES_CM:
            t_block, dist_mean, _ = multi_trial_block_average(grams, d_cm)
            exp_data[(grams, d_cm)] = (t_block, dist_mean - d_cm)
    return exp_data


def fit_params():
    """Fit IR observation model; returns OptimizeResult and prints source-code-ready values."""
    beta_cache = _precompute_optics()
    exp_data = _load_experiment()

    def objective(params):
        log10_eta, log10_scale_lo, log10_scale_hi, x_reg_m, x_overlap_m = params
        eta = 10.0 ** log10_eta
        scale_map = {0.2: 10.0 ** log10_scale_lo,
                     0.8: 10.0 ** log10_scale_hi}
        total_sse = 0.0
        n_total = 0

        for grams in FIT_GRAMS:
            optics = beta_cache[grams]
            scale = scale_map[grams]
            alpha = optics['alpha_ext_1_per_m'] * scale
            beta = optics['beta_back_1_per_m_sr'] * scale
            t_model = optics['t_s']

            for d_cm in FIT_DISTANCES:
                d_m = d_cm / 100.0
                d_est = dust_biased_distance(
                    d_m, alpha, beta, eta=eta,
                    x_reg_m=x_reg_m, x_overlap_m=x_overlap_m)
                err_model = d_est * 100.0 - d_cm

                t_exp, err_exp = exp_data[(grams, d_cm)]
                if len(t_exp) == 0:
                    continue
                err_model_interp = np.interp(t_exp, t_model, err_model)

                scale_err = max(np.nanmax(np.abs(err_exp)), 0.1)
                valid = np.isfinite(err_exp) & np.isfinite(err_model_interp)
                total_sse += np.sum(((err_exp[valid] - err_model_interp[valid]) /
                                     scale_err) ** 2)
                n_total += int(valid.sum())
        return total_sse / max(n_total, 1)

    print('Fitting corrected IR parameters ...')
    bounds = [
        (-8.0, 4.0),
        (-2.0, 2.0),
        (-2.0, 2.0),
        (0.002, 0.12),
        (0.005, 0.30),
    ]
    result = differential_evolution(
        objective, bounds, seed=42, maxiter=300, tol=1e-8, polish=True)

    log10_eta, log10_slo, log10_shi, x_reg, x_overlap = result.x
    eta = 10.0 ** log10_eta
    scale_lo = 10.0 ** log10_slo
    scale_hi = 10.0 ** log10_shi
    rmse_norm = np.sqrt(result.fun)

    print(f'\n{"=" * 60}')
    print('Optimal corrected IR parameters:')
    print(f'  ETA_BACKSCATTER = {eta:.8g}')
    print(f'  CONCENTRATION_SCALE[0.2] = {scale_lo:.8g}')
    print(f'  CONCENTRATION_SCALE[0.8] = {scale_hi:.8g}')
    print(f'  X_REG_M = {x_reg:.6f}')
    print(f'  X_OVERLAP_M = {x_overlap:.6f}')
    print(f'  normalized MSE  = {result.fun:.6f}')
    print(f'  normalized RMSE = {rmse_norm:.4f}')
    print(f'  previous ETA initial guess was {ETA_BACKSCATTER:.8g}; '
          f'previous x_reg/x_overlap were {X_REG_M:.4f}/{X_OVERLAP_M:.4f} m')
    print(f'{"=" * 60}')

    return result


if __name__ == '__main__':
    fit_params()
