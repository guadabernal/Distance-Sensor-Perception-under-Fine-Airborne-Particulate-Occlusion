"""Clean-air baselines: LUT calibration transfer to the chamber and within-run temporal precision."""

import numpy as np
import pandas as pd
from scipy import stats

from calibration_analysis import ir_model, vl5_calibration_cm
from run_level_metrics import CENTER_CELLS, CENTER_RAW_COLS, CENTER_CM_COLS


CONTROL_WINDOW_S = 20.0
CONTROL_MARGIN_S = 5.0


def control_window(control):
    """Stable window inside the control phase, matching the preprocessing bias offset."""
    t = control['time_seconds']
    win_end = t.max() - CONTROL_MARGIN_S
    return control[(t >= win_end - CONTROL_WINDOW_S) & (t <= win_end)]


def _baseline_ir(window, fits):
    pre_offset = float(np.nanmedian(
        ir_model(window['voltage_filtered'].values, fits['ir_A'], fits['ir_B'])))
    d = window['distance_cm_filtered'].values
    return pre_offset, {'temporal_precision_cm': float(np.std(d[np.isfinite(d)], ddof=1))}


def _baseline_vl5(window, fits, D):
    zone_pre_means = np.nanmean(vl5_calibration_cm(window[CENTER_RAW_COLS].values, fits), axis=0)
    cm = window[CENTER_CM_COLS].values
    zone_sds = np.array([np.std(cm[np.isfinite(cm[:, z]), z], ddof=1)
                         for z in range(cm.shape[1])])
    row = {
        'temporal_precision_cm': float(np.mean(zone_sds)),
        'zone_temporal_sd_min_cm': float(np.min(zone_sds)),
        'zone_temporal_sd_max_cm': float(np.max(zone_sds)),
        'zone_mean_spatial_sd_cm': float(np.std(np.nanmean(cm, axis=0), ddof=1)),
    }
    for cell, sd, mean in zip(CENTER_CELLS, zone_sds, zone_pre_means):
        row[f'zone_{cell}_temporal_sd_cm'] = float(sd)
        row[f'zone_{cell}_mean_error_cm'] = float(mean - D)
    return float(np.mean(zone_pre_means)), row


def build_baseline_trial_metrics(runs, fits):
    """One row per run (72): pre-offset LUT calibration error and clean-air precision."""
    rows = []
    for (sensor, grams, dist_cm, trial), (control, _) in runs.items():
        window = control_window(control)
        if sensor == 'IR':
            pre_offset, extra = _baseline_ir(window, fits)
        else:
            pre_offset, extra = _baseline_vl5(window, fits, float(dist_cm))
        bias = pre_offset - dist_cm
        rows.append({
            'sensor_code': sensor, 'loading_label_g': grams, 'trial': trial,
            'distance_cm': dist_cm, 'n_control_frames': len(window),
            'pre_offset_estimate_cm': pre_offset,
            'calibration_bias_cm': bias, 'calibration_abs_error_cm': abs(bias),
            **extra,
        })
    return pd.DataFrame(rows)


def build_baseline_summary(trial_df):
    """Six baseline runs per (sensor, distance): calibration transfer error and precision spread."""
    rows = []
    for (sensor, dist_cm), grp in trial_df.groupby(['sensor_code', 'distance_cm'], sort=True):
        n = len(grp)
        e = grp['calibration_bias_cm'].values
        mean = float(np.mean(e))
        sd = float(np.std(e, ddof=1))
        sem = sd / np.sqrt(n)
        tq = float(stats.t.ppf(0.975, n - 1))
        p = grp['temporal_precision_cm'].values
        rows.append({
            'sensor_code': sensor, 'distance_cm': dist_cm, 'n_baseline_runs': n,
            'calibration_bias_mean_cm': mean,
            'calibration_bias_sd_cm': sd,
            'calibration_bias_sem_cm': sem,
            'calibration_bias_ci95_low_cm': mean - tq * sem,
            'calibration_bias_ci95_high_cm': mean + tq * sem,
            'calibration_transfer_rmse_cm': float(np.sqrt(np.mean(e ** 2))),
            'calibration_transfer_mae_cm': float(np.mean(np.abs(e))),
            'calibration_abs_error_min_cm': float(np.min(np.abs(e))),
            'calibration_abs_error_max_cm': float(np.max(np.abs(e))),
            'precision_mean_cm': float(np.mean(p)),
            'precision_median_cm': float(np.median(p)),
            'precision_sd_across_runs_cm': float(np.std(p, ddof=1)),
            'precision_min_cm': float(np.min(p)),
            'precision_max_cm': float(np.max(p)),
        })
    return pd.DataFrame(rows)
