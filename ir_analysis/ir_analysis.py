"""IR sensor data loading, block averaging, multi-trial metrics, ANOVA, and paper summary."""

import os
import numpy as np
import pandas as pd

_trapz = getattr(np, 'trapezoid', None) or np.trapz
from scipy.optimize import curve_fit

from ir_model import simulate_experiment


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data',
                        'processed_experiments', 'IR')

DISTANCES_CM = [10, 20, 30, 40, 50, 60]
DUST_LOADINGS_G = [0.2, 0.8]
TRIALS = [1, 2, 3]
BLOCK_SIZE = 83


def load_trial_data(grams, distance_cm, trial):
    """Load a single trial's processed CSV."""
    filename = f'IR_{grams}g_TRIAL{trial}_{distance_cm}cm.csv'
    return pd.read_csv(os.path.join(DATA_DIR, filename))


def load_trial1_data(grams, distance_cm):
    """Load TRIAL1 processed CSV."""
    return load_trial_data(grams, distance_cm, 1)


def phase_time_seconds(df_phase):
    """Return phase-local time [s], rebased to first sample (robust to global/per-phase convention)."""
    t = np.asarray(df_phase['time_seconds'].values, dtype=float)
    if len(t) == 0:
        return t
    return t - t[0]


def block_average(time_s, values, block_size=BLOCK_SIZE):
    """Mean of every block_size consecutive samples; returns (block_times, block_means)."""
    n_blocks = len(values) // block_size
    if n_blocks == 0:
        return np.array([]), np.array([])

    t = np.asarray(time_s[:n_blocks * block_size]).reshape(n_blocks, block_size)
    v = np.asarray(values[:n_blocks * block_size]).reshape(n_blocks, block_size)

    return t.mean(axis=1), v.mean(axis=1)


def multi_trial_block_average(grams, distance_cm, phase='measurement'):
    """Block-average each valid trial then aggregate across trials; returns (t, mean, std)."""
    block_dists = []
    block_time = None
    for trial in TRIALS:
        df = load_trial_data(grams, distance_cm, trial)
        sub = df[df['phase_type'] == phase]
        t_blk, d_blk = block_average(
            phase_time_seconds(sub),
            sub['distance_cm_smooth'].values,
        )
        block_dists.append(d_blk)
        if block_time is None:
            block_time = t_blk

    min_len = min(len(b) for b in block_dists)
    block_time = block_time[:min_len]
    stacked = np.column_stack([b[:min_len] for b in block_dists])

    return block_time, stacked.mean(axis=1), stacked.std(axis=1)


def _is_valid(dist_cm):
    """Boolean mask: True where reading is finite, positive, and <= 100 cm."""
    d = np.asarray(dist_cm, dtype=float)
    return np.isfinite(d) & (d > 0) & (d <= 100)


def _recovery_time(block_times, block_errors, tolerance_cm, dwell_s=10.0):
    """First block time [s] where |error| stays within tolerance for dwell_s, else NaN."""
    if len(block_times) < 2:
        return np.nan
    within = np.abs(block_errors) <= tolerance_cm
    dt_block = np.median(np.diff(block_times))
    n_dwell = max(1, int(np.ceil(dwell_s / dt_block)))
    for i in range(len(within) - n_dwell + 1):
        if np.all(within[i:i + n_dwell]):
            return block_times[i]
    return np.nan


def fit_exponential_decay(block_time_s, error_cm):
    """Fit error(t) = A*exp(-t/tau) + C; returns {A, tau_s, C, r_squared} or all-NaN."""
    t = np.asarray(block_time_s, dtype=float)
    e = np.asarray(error_cm, dtype=float)
    nan_result = {'A': np.nan, 'tau_s': np.nan, 'C': np.nan, 'r_squared': np.nan}

    if len(t) < 4 or np.ptp(e) < 1e-6:
        return nan_result

    def model(t, A, tau, C):
        return A * np.exp(-t / tau) + C

    try:
        popt, _ = curve_fit(
            model, t, e,
            p0=[e[0] - e[-1], 120.0, e[-1]],
            bounds=([-np.inf, 1e-3, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=5000,
        )
        pred = model(t, *popt)
        ss_res = np.sum((e - pred) ** 2)
        ss_tot = np.sum((e - e.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {'A': popt[0], 'tau_s': popt[1], 'C': popt[2], 'r_squared': r_sq}
    except (RuntimeError, ValueError):
        return nan_result


def fit_error_vs_distance(distances_cm, errors_cm):
    """Fit |error| = a*d^n via log-log regression; returns {a, n, r_squared}."""
    d = np.asarray(distances_cm, dtype=float)
    e = np.abs(np.asarray(errors_cm, dtype=float))

    valid = (d > 0) & (e > 0) & np.isfinite(e)
    if valid.sum() < 2:
        return {'a': np.nan, 'n': np.nan, 'r_squared': np.nan}

    log_d = np.log(d[valid])
    log_e = np.log(e[valid])

    coeffs = np.polyfit(log_d, log_e, 1)
    n = coeffs[0]
    a = np.exp(coeffs[1])

    pred = np.polyval(coeffs, log_d)
    ss_res = np.sum((log_e - pred) ** 2)
    ss_tot = np.sum((log_e - log_e.mean()) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {'a': a, 'n': n, 'r_squared': r_sq}


def compute_run_metrics_multitrial(grams, d_cm):
    """Compute all metrics on 3-trial mean block-averaged data."""
    block_time, mean_dist, std_dist = multi_trial_block_average(grams, d_cm)
    error = mean_dist - d_cm
    abs_err = np.abs(error)

    m = {'grams': grams, 'distance_cm': d_cm}

    m['inter_trial_mean_std'] = std_dist.mean()
    m['cv'] = std_dist.mean() / d_cm

    ctrl_time, ctrl_mean, ctrl_std = multi_trial_block_average(
        grams, d_cm, phase='control')
    if len(ctrl_mean) > 0:
        m['baseline_mean'] = ctrl_mean.mean()
        m['baseline_bias'] = ctrl_mean.mean() - d_cm
        m['baseline_std'] = ctrl_std.mean()
        valid_fracs = []
        for trial in TRIALS:
            df = load_trial_data(grams, d_cm, trial)
            ctrl = df[df['phase_type'] == 'control']['distance_cm_smooth'].values
            valid_fracs.append(_is_valid(ctrl).mean() if len(ctrl) > 0 else np.nan)
        m['baseline_valid_frac'] = np.nanmean(valid_fracs)
    else:
        m['baseline_mean'] = m['baseline_bias'] = m['baseline_std'] = np.nan
        m['baseline_valid_frac'] = np.nan

    if len(error) > 0:
        m['bias_full'] = error.mean()

        early = block_time <= 60.0
        m['bias_early_60s'] = error[early].mean() if early.any() else np.nan

        t_max = block_time.max()
        late = block_time >= (t_max - 120.0)
        m['bias_late_120s'] = error[late].mean() if late.any() else np.nan

        m['rmse_full'] = np.sqrt((error ** 2).mean())
        m['mae_full'] = abs_err.mean()
        m['medae_full'] = np.median(abs_err)
        m['p90_abs_error'] = np.percentile(abs_err, 90)
        m['p95_abs_error'] = np.percentile(abs_err, 95)
        m['p99_abs_error'] = np.percentile(abs_err, 99)
        m['nrmse'] = m['rmse_full'] / d_cm

        m['early_mae_60s'] = (
            np.abs(error[early]).mean() if early.any() else np.nan
        )
        m['late_mae_120s'] = (
            np.abs(error[late]).mean() if late.any() else np.nan
        )

        m['peak_abs_error'] = abs_err.max()
        peak_idx = int(np.argmax(abs_err))
        m['time_to_peak_error_s'] = block_time[peak_idx]
        m['peak_error_pct'] = abs_err.max() / d_cm * 100

        m['recovery_time_5pct_s'] = _recovery_time(
            block_time, error, 0.05 * d_cm)
        m['recovery_time_10pct_s'] = _recovery_time(
            block_time, error, 0.10 * d_cm)

        m['frac_impaired_5pct'] = (abs_err > 0.05 * d_cm).mean()
        m['frac_impaired_10pct'] = (abs_err > 0.10 * d_cm).mean()

        m['iae'] = _trapz(abs_err, block_time)
        m['ise'] = _trapz(error ** 2, block_time)
    else:
        for k in [
            'bias_full', 'bias_early_60s', 'bias_late_120s',
            'rmse_full', 'mae_full', 'medae_full',
            'p90_abs_error', 'p95_abs_error', 'p99_abs_error',
            'nrmse', 'early_mae_60s', 'late_mae_120s',
            'peak_abs_error', 'time_to_peak_error_s', 'peak_error_pct',
            'recovery_time_5pct_s', 'recovery_time_10pct_s',
            'frac_impaired_5pct', 'frac_impaired_10pct',
            'iae', 'ise',
        ]:
            m[k] = np.nan

    valid_fracs, blind_times, first_valids = [], [], []
    for trial in TRIALS:
        df = load_trial_data(grams, d_cm, trial)
        meas = df[df['phase_type'] == 'measurement']
        meas_dist = meas['distance_cm_smooth'].values
        meas_time = phase_time_seconds(meas)
        meas_valid = _is_valid(meas_dist)
        if len(meas_dist) > 0:
            valid_fracs.append(meas_valid.mean())
            dur = meas_time[-1] - meas_time[0] if len(meas_time) > 1 else 0.0
            blind_times.append((1 - meas_valid.mean()) * dur)
            vi = np.where(meas_valid)[0]
            first_valids.append(meas_time[vi[0]] if len(vi) > 0 else np.nan)
        else:
            valid_fracs.append(np.nan)
            blind_times.append(np.nan)
            first_valids.append(np.nan)
    m['valid_frac_full'] = np.nanmean(valid_fracs)
    m['blind_time_s'] = np.nanmean(blind_times)
    m['time_to_first_valid_s'] = np.nanmean(first_valids)

    if len(error) > 0:
        decay = fit_exponential_decay(block_time, error)
        m['decay_tau_s'] = decay['tau_s']
        m['decay_A'] = decay['A']
        m['decay_C'] = decay['C']
        m['decay_r_squared'] = decay['r_squared']
    else:
        m['decay_tau_s'] = m['decay_A'] = m['decay_C'] = np.nan
        m['decay_r_squared'] = np.nan

    sim = simulate_experiment(distance_cm=d_cm, mass_g=grams)

    if len(error) > 0:
        model_err = np.interp(block_time, sim['t_s'],
                              sim['distance_est_cm'] - d_cm)
        residual = error - model_err

        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum((error - error.mean()) ** 2)

        m['residual_rmse'] = np.sqrt((residual ** 2).mean())
        m['residual_mae'] = np.abs(residual).mean()
        m['residual_bias'] = residual.mean()
        m['r_squared'] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        if len(block_time) > 1:
            cc = np.corrcoef(error, model_err)
            m['correlation'] = cc[0, 1]
        else:
            m['correlation'] = np.nan

        early = block_time <= 60.0
        m['early_residual_rmse_60s'] = (
            np.sqrt((residual[early] ** 2).mean()) if early.any() else np.nan
        )

        late = block_time >= (block_time.max() - 120.0)
        m['late_residual_rmse_120s'] = (
            np.sqrt((residual[late] ** 2).mean()) if late.any() else np.nan
        )
    else:
        for k in [
            'residual_rmse', 'residual_mae', 'residual_bias',
            'r_squared', 'correlation',
            'early_residual_rmse_60s', 'late_residual_rmse_120s',
        ]:
            m[k] = np.nan

    return m


def compute_all_metrics():
    """Compute multi-trial metrics for all conditions. Returns DataFrame."""
    rows = []
    for grams in DUST_LOADINGS_G:
        for d_cm in DISTANCES_CM:
            print(f'  Computing metrics: {grams}g, {d_cm} cm ...')
            rows.append(compute_run_metrics_multitrial(grams, d_cm))
    return pd.DataFrame(rows)


def compute_anova():
    """Two-way ANOVA on per-trial MAE: MAE ~ grams * distance_cm."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols as sm_ols

    rows = []
    for grams in DUST_LOADINGS_G:
        for d_cm in DISTANCES_CM:
            for trial in TRIALS:
                df = load_trial_data(grams, d_cm, trial)
                meas = df[df['phase_type'] == 'measurement']
                dist = meas['distance_cm_smooth'].values
                valid = _is_valid(dist)
                mae = np.abs(dist[valid] - d_cm).mean() if valid.any() else np.nan
                rows.append({
                    'grams': grams, 'distance_cm': d_cm,
                    'trial': trial, 'mae': mae,
                })

    df_trials = pd.DataFrame(rows).dropna()

    model = sm_ols('mae ~ C(grams) * C(distance_cm)', data=df_trials).fit()
    return sm.stats.anova_lm(model, typ=2)


def _pooled_r_squared(loadings):
    """Pooled R² of the model fit across (loading, distance) conditions."""
    errs, models = [], []
    for grams in loadings:
        for d_cm in DISTANCES_CM:
            t_block, dist_mean, _ = multi_trial_block_average(grams, d_cm)
            error = dist_mean - d_cm
            if len(error) == 0:
                continue
            sim = simulate_experiment(distance_cm=d_cm, mass_g=grams)
            model_err = np.interp(t_block, sim['t_s'],
                                  sim['distance_est_cm'] - d_cm)
            errs.append(error)
            models.append(model_err)
    if not errs:
        return np.nan
    p_err = np.concatenate(errs)
    p_mod = np.concatenate(models)
    ss_res = np.sum((p_err - p_mod) ** 2)
    ss_tot = np.sum((p_err - p_err.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def save_paper_summary(df_metrics, anova_table, save_path):
    """Write key aggregate metrics for the paper's results section as CSV."""
    rows = []

    peak = df_metrics[['grams', 'distance_cm', 'peak_abs_error']].dropna()
    if len(peak) > 0:
        row_min = peak.loc[peak['peak_abs_error'].idxmin()]
        row_max = peak.loc[peak['peak_abs_error'].idxmax()]
        rows += [
            ('peak_abs_error_min_cm', round(row_min['peak_abs_error'], 3)),
            ('peak_abs_error_min_at', f"{int(row_min['distance_cm'])}cm_{row_min['grams']}g"),
            ('peak_abs_error_max_cm', round(row_max['peak_abs_error'], 3)),
            ('peak_abs_error_max_at', f"{int(row_max['distance_cm'])}cm_{row_max['grams']}g"),
        ]

    rec5 = df_metrics['recovery_time_5pct_s'].dropna()
    if len(rec5) > 0:
        rows += [('recovery_5pct_min_s', round(rec5.min(), 2)),
                 ('recovery_5pct_max_s', round(rec5.max(), 2))]

    r2 = df_metrics['r_squared'].dropna()
    if len(r2) > 0:
        rows += [('r_squared_min', round(r2.min(), 4)),
                 ('r_squared_max', round(r2.max(), 4)),
                 ('r_squared_mean', round(r2.mean(), 4))]

    rows.append(('r_squared_pooled_global',
                 round(_pooled_r_squared(DUST_LOADINGS_G), 4)))
    for grams in DUST_LOADINGS_G:
        rows.append((f'r_squared_pooled_{grams}g',
                     round(_pooled_r_squared([grams]), 4)))

    tau = df_metrics['decay_tau_s'].dropna()
    if len(tau) > 0:
        rows += [('decay_tau_min_s', round(tau.min(), 2)),
                 ('decay_tau_max_s', round(tau.max(), 2))]

    if anova_table is not None:
        name_map = {'C(grams)': 'grams', 'C(distance_cm)': 'distance',
                    'C(grams):C(distance_cm)': 'interaction'}
        for source in anova_table.index:
            if source == 'Residual':
                continue
            short = name_map.get(source, source)
            row = anova_table.loc[source]
            rows += [(f'anova_{short}_F', round(float(row['F']), 3)),
                     (f'anova_{short}_p', round(float(row['PR(>F)']), 6))]

    pd.DataFrame(rows, columns=['metric', 'value']).to_csv(save_path, index=False)
    print(f'Saved: {save_path}')
