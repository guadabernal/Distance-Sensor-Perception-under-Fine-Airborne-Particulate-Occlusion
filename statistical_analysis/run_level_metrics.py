"""Run-level dust-response metrics (MAE, nMAE, bias, RMSE, false lock) per window and condition summaries."""

import os
import numpy as np
import pandas as pd
from scipy import stats

_trapz = getattr(np, 'trapezoid', None) or np.trapz


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data', 'processed_experiments')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics')

SENSORS = ['IR', 'VL5']
LOADINGS_G = [0.2, 0.8]
DISTANCES_CM = [10, 20, 30, 40, 50, 60]
TRIALS = [1, 2, 3]

CENTER_CELLS = [27, 28, 35, 36]
CENTER_RAW_COLS = [f'cell_{i}' for i in CENTER_CELLS]
CENTER_CM_COLS = [f'cell_{i}_cm' for i in CENTER_CELLS]
CENTER_FL_COLS = [f'cell_{i}_fl' for i in CENTER_CELLS]

WINDOWS = {'full_15min': (0.0, 900.0), 'initial_30s': (0.0, 30.0)}


def load_run(sensor, grams, trial, dist_cm):
    """Processed run split into (control, measurement) phases."""
    filename = f'{sensor}_{grams}g_TRIAL{trial}_{dist_cm}cm.csv'
    df = pd.read_csv(os.path.join(DATA_DIR, sensor, filename))
    return df[df['phase_type'] == 'control'], df[df['phase_type'] == 'measurement']


def load_all_runs():
    """All 72 processed runs keyed by (sensor, grams, dist_cm, trial)."""
    runs = {}
    for sensor in SENSORS:
        for grams in LOADINGS_G:
            for dist_cm in DISTANCES_CM:
                for trial in TRIALS:
                    runs[(sensor, grams, dist_cm, trial)] = load_run(
                        sensor, grams, trial, dist_cm)
    return runs


def time_weighted_mean(t, q, start, end):
    """Trapezoidal time average of q(t) over [start, end], interpolating the window edges."""
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    inside = (t >= start) & (t <= end)
    t_c, q_c = t[inside], q[inside]
    if len(t_c) and t_c[0] > start and (t < start).any():
        t_c = np.insert(t_c, 0, start)
        q_c = np.insert(q_c, 0, np.interp(start, t, q))
    if len(t_c) and t_c[-1] < end and (t > end).any():
        t_c = np.append(t_c, end)
        q_c = np.append(q_c, np.interp(end, t, q))
    if len(t_c) == 0:
        return np.nan
    if len(t_c) == 1 or t_c[-1] == t_c[0]:
        return float(q_c[0])
    return float(_trapz(q_c, t_c) / (t_c[-1] - t_c[0]))


def window_metrics(t, dist, true_cm, start, end, fl=None):
    """Time-weighted error metrics for one run over [start, end]; dist is (frames, zones)."""
    D = float(true_cm)
    err = np.asarray(dist, dtype=float) - D
    finite = np.isfinite(err)
    any_finite = finite.any(axis=1)
    err_f = np.where(finite, err, np.nan)
    with np.errstate(invalid='ignore'):
        signed = np.where(any_finite, np.nanmean(err_f, axis=1), np.nan)
        abs_err = np.where(any_finite, np.nanmean(np.abs(err_f), axis=1), np.nan)
        sq_err = np.where(any_finite, np.nanmean(err_f ** 2, axis=1), np.nan)
    gt10 = (finite & (np.abs(err) > 0.10 * D)).mean(axis=1)

    def tw(values):
        keep = np.isfinite(values)
        return time_weighted_mean(t[keep], values[keep], start, end)

    in_win = (t >= start) & (t <= end)
    mae = tw(abs_err)
    bias = tw(signed)
    msq = tw(sq_err)
    return {
        'n_frames': int(in_win.sum()),
        'valid_fraction': float(finite[in_win].mean()),
        'mae_cm': mae,
        'nmae_pct': 100.0 * mae / D,
        'bias_cm': bias,
        'bias_pct': 100.0 * bias / D,
        'rmse_cm': float(np.sqrt(msq)),
        'frac_abs_error_gt_10pct_time_weighted': time_weighted_mean(t, gt10, start, end),
        'false_lock_fraction': tw(fl.mean(axis=1)) if fl is not None else np.nan,
    }


def build_run_level_metrics(runs):
    """One row per run and window (144 rows); IR uses distance_cm_filtered, dToF the center 2x2 zones."""
    rows = []
    for (sensor, grams, dist_cm, trial), (_, meas) in runs.items():
        t = meas['time_seconds'].values
        if sensor == 'IR':
            dist, fl = meas[['distance_cm_filtered']].values, None
        else:
            dist = meas[CENTER_CM_COLS].values
            fl = meas[CENTER_FL_COLS].values.astype(float)
        for window, (a, b) in WINDOWS.items():
            rows.append({
                'sensor_code': sensor, 'loading_g': grams,
                'distance_cm': dist_cm, 'trial': trial, 'window': window,
                **window_metrics(t, dist, dist_cm, a, b, fl=fl),
            })
    return pd.DataFrame(rows)


def _t_ci(values):
    """Mean, sample SD, SEM and Student-t 95% CI."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    sem = sd / np.sqrt(n)
    tq = float(stats.t.ppf(0.975, n - 1))
    return mean, sd, sem, mean - tq * sem, mean + tq * sem


def summarize_conditions(run_df):
    """Three runs per (sensor, loading, distance, window): Student-t CIs, geometric-mean nMAE, false lock."""
    metric_specs = [
        ('nmae', 'nmae_pct', '_pct'),
        ('bias', 'bias_cm', '_cm'),
        ('mae', 'mae_cm', '_cm'),
        ('rmse', 'rmse_cm', '_cm'),
        ('valid_fraction', 'valid_fraction', ''),
        ('frac_abs_error_gt_10pct', 'frac_abs_error_gt_10pct_time_weighted', ''),
    ]
    rows = []
    group_cols = ['sensor_code', 'loading_g', 'distance_cm', 'window']
    for (sensor, grams, dist_cm, window), grp in run_df.groupby(group_cols, sort=True):
        row = {'sensor_code': sensor, 'loading_g': grams, 'distance_cm': dist_cm,
               'window': window, 'n_runs': len(grp)}
        for name, col, suffix in metric_specs:
            mean, sd, sem, lo, hi = _t_ci(grp[col].values)
            row[f'{name}_mean{suffix}'] = mean
            row[f'{name}_sd{suffix}'] = sd
            row[f'{name}_sem{suffix}'] = sem
            row[f'{name}_ci95_low{suffix}'] = lo
            row[f'{name}_ci95_high{suffix}'] = hi

        gmean, _, _, glo, ghi = _t_ci(np.log(grp['nmae_pct'].values))
        row['nmae_geomean_pct'] = float(np.exp(gmean))
        row['nmae_geomean_ci95_low_pct'] = float(np.exp(glo))
        row['nmae_geomean_ci95_high_pct'] = float(np.exp(ghi))

        fl = grp['false_lock_fraction'].values
        is_vl5 = sensor == 'VL5'
        row['false_lock_mean'] = float(np.mean(fl)) if is_vl5 else np.nan
        row['false_lock_sd'] = float(np.std(fl, ddof=1)) if is_vl5 else np.nan
        row['false_lock_min'] = float(np.min(fl)) if is_vl5 else np.nan
        row['false_lock_max'] = float(np.max(fl)) if is_vl5 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
