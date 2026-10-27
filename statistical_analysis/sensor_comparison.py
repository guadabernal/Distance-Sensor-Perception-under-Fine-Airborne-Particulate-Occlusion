"""Sensor comparison on run-level log nMAE: HC3 factorial model, per-condition Welch contrasts, and a permutation test."""

import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from run_level_metrics import LOADINGS_G, DISTANCES_CM, WINDOWS


FORMULA = ('log_nmae ~ C(sensor_code, Sum) * C(loading_g, Sum) '
           '* C(distance_cm, Sum)')

TERM_LABELS = {
    'C(sensor_code, Sum)': 'sensor',
    'C(loading_g, Sum)': 'loading',
    'C(distance_cm, Sum)': 'distance',
    'C(sensor_code, Sum):C(loading_g, Sum)': 'sensor x loading',
    'C(sensor_code, Sum):C(distance_cm, Sum)': 'sensor x distance',
    'C(loading_g, Sum):C(distance_cm, Sum)': 'loading x distance',
    'C(sensor_code, Sum):C(loading_g, Sum):C(distance_cm, Sum)':
        'sensor x loading x distance',
}

PERMUTATION_COUNT = 100_000
RANDOM_SEED = 20260819


def _model_frame(run_df, window):
    sub = run_df[run_df['window'] == window].copy()
    sub['log_nmae'] = np.log(sub['nmae_pct'].values)
    return sub.sort_values(['sensor_code', 'loading_g', 'distance_cm', 'trial'],
                           kind='stable').reset_index(drop=True)


def build_factorial_tests(run_df):
    """Robust (HC3) Wald tests for every term of the full sensor x loading x distance model."""
    rows = []
    for window in WINDOWS:
        result = smf.ols(FORMULA, data=_model_frame(run_df, window)).fit(
            cov_type='HC3', use_t=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            wald = result.wald_test_terms(skip_single=False)
        table = wald.table
        is_f = 'df_denom' in table.columns and str(wald.distribution).lower() != 'chi2'
        for term, label in TERM_LABELS.items():
            rec = table.loc[term]
            rows.append({
                'window': window, 'term': label,
                'statistic_name': 'robust_wald_F' if is_f else 'robust_wald_chi2',
                'statistic': float(np.squeeze(rec['statistic'])),
                'df_num': float(rec['df_constraint']),
                'df_denom': float(rec['df_denom']) if is_f else np.nan,
                'p_value': float(np.squeeze(rec['pvalue'])),
                'n_observations': int(result.nobs),
                'residual_df': float(result.df_resid),
                'model_r_squared': float(result.rsquared),
                'model_adjusted_r_squared': float(result.rsquared_adj),
            })
    return pd.DataFrame(rows)


def welch_log_contrast(y_vl5_log, y_ir_log):
    """Welch-Satterthwaite t contrast of mean log nMAE (dToF - IR)."""
    y1 = np.asarray(y_vl5_log, dtype=float)
    y2 = np.asarray(y_ir_log, dtype=float)
    n1, n2 = len(y1), len(y2)
    delta = float(np.mean(y1) - np.mean(y2))
    v1 = float(np.var(y1, ddof=1)) / n1
    v2 = float(np.var(y2, ddof=1)) / n2
    se = float(np.sqrt(v1 + v2))
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    t_stat = delta / se
    p = 2.0 * float(stats.t.sf(abs(t_stat), df))
    tq = float(stats.t.ppf(0.975, df))
    return delta, se, t_stat, float(df), p, delta - tq * se, delta + tq * se


def build_sensor_contrasts(run_df):
    """GM nMAE ratio dToF/IR per condition with log-scale CIs and Holm correction within window."""
    rows = []
    for window in WINDOWS:
        frame = _model_frame(run_df, window)
        window_rows = []
        for grams in LOADINGS_G:
            for dist_cm in DISTANCES_CM:
                cell = frame[(frame['loading_g'] == grams) & (frame['distance_cm'] == dist_cm)]
                y_vl5 = cell[cell['sensor_code'] == 'VL5']['log_nmae'].values
                y_ir = cell[cell['sensor_code'] == 'IR']['log_nmae'].values
                delta, se, t_stat, df_ws, p_raw, lo, hi = welch_log_contrast(y_vl5, y_ir)
                ratio = float(np.exp(delta))
                if ratio < 1.0:
                    lower = 'dToF sensor'
                elif ratio > 1.0:
                    lower = 'IR sensor'
                else:
                    lower = 'equal'
                window_rows.append({
                    'window': window, 'loading_g': grams, 'distance_cm': dist_cm,
                    'log_ratio_estimate': delta, 'log_ratio_se': se,
                    'test_statistic': t_stat, 'df_satterthwaite': df_ws,
                    'ratio_dtof_over_ir': ratio,
                    'ratio_ci95_low': float(np.exp(lo)),
                    'ratio_ci95_high': float(np.exp(hi)),
                    'p_value_raw': p_raw,
                    'lower_error_sensor': lower,
                    'n_ir_runs': len(y_ir), 'n_dtof_runs': len(y_vl5),
                })
        _, p_holm, _, _ = multipletests([r['p_value_raw'] for r in window_rows],
                                        alpha=0.05, method='holm')
        for r, p in zip(window_rows, p_holm):
            r['p_value_holm'] = float(p)
        rows.extend(window_rows)
    return pd.DataFrame(rows)


def permutation_test(frame, n_permutations, seed, chunk_size=20_000):
    """Partial F for sensor within loading x distance strata; labels permuted within strata."""
    y_ir, y_vl5 = [], []
    for grams in LOADINGS_G:
        for dist_cm in DISTANCES_CM:
            cell = frame[(frame['loading_g'] == grams) & (frame['distance_cm'] == dist_cm)]
            y_ir.append(cell[cell['sensor_code'] == 'IR']['log_nmae'].values)
            y_vl5.append(cell[cell['sensor_code'] == 'VL5']['log_nmae'].values)
    y_all = np.concatenate([np.array(y_ir), np.array(y_vl5)], axis=1)
    n_strata, per_stratum = y_all.shape
    df_num = n_strata
    df_denom = y_all.size - 2 * n_strata
    sse_reduced = float(np.sum((y_all - y_all.mean(axis=1, keepdims=True)) ** 2))
    delta = float(np.sum(1.5 * (np.array(y_ir).mean(axis=1)
                                - np.array(y_vl5).mean(axis=1)) ** 2))
    f_obs = (delta / df_num) / ((sse_reduced - delta) / df_denom)

    total = y_all.sum(axis=1)
    rng = np.random.default_rng(seed)
    f_perm = np.empty(n_permutations)
    done = 0
    while done < n_permutations:
        b = min(chunk_size, n_permutations - done)
        order = rng.random((b, n_strata, per_stratum)).argsort(axis=2)
        y_bc = np.broadcast_to(y_all, (b, n_strata, per_stratum))
        sum_ir = np.take_along_axis(y_bc, order[:, :, :3], axis=2).sum(axis=2)
        delta_b = 1.5 * ((sum_ir / 3.0 - (total[None, :] - sum_ir) / 3.0) ** 2).sum(axis=1)
        f_perm[done:done + b] = (delta_b / df_num) / ((sse_reduced - delta_b) / df_denom)
        done += b

    n_extreme = int(np.sum(f_perm >= f_obs))
    return {
        'observed_partial_f': f_obs, 'df_num': df_num, 'df_denom': df_denom,
        'n_permutations': n_permutations, 'n_extreme': n_extreme,
        'p_value': (n_extreme + 1) / (n_permutations + 1),
    }


def build_permutation_tests(run_df, n_permutations=PERMUTATION_COUNT, seed=RANDOM_SEED):
    rows = []
    for wi, window in enumerate(WINDOWS):
        frame = _model_frame(run_df, window)
        rows.append({'window': window,
                     **permutation_test(frame, n_permutations, seed=[seed, wi]),
                     'random_seed': seed, 'n_observations': len(frame)})
    return pd.DataFrame(rows)
