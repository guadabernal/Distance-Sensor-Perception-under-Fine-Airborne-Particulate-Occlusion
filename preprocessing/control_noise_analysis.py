"""Control-phase noise (SD) analysis for IR and VL5 trials over the bias-averaging window."""

import os
import glob
import re

import numpy as np
import pandas as pd

from data_pre_processing import (
    PROCESSED_DIR,
    CONTROL_WINDOW_S,
    CONTROL_MARGIN_S,
    VL5_CENTER_2X2,
)

IR_DIR = os.path.join(PROCESSED_DIR, 'IR')
VL5_DIR = os.path.join(PROCESSED_DIR, 'VL5')
DISTANCES_CM = [10, 20, 30, 40, 50, 60]

IR_NOISE_COL = 'distance_cm_filtered'
VL5_CENTER_CM_COLS = [f'cell_{ci}_cm' for ci in VL5_CENTER_2X2]


_FNAME_RE = re.compile(r'(IR|VL5)_(\d+\.?\d*)g_TRIAL(\d+)_(\d+)cm\.csv$')


def parse_filename(path):
    m = _FNAME_RE.search(os.path.basename(path))
    if not m:
        return None
    return {
        'sensor': m.group(1),
        'grams': float(m.group(2)),
        'trial': int(m.group(3)),
        'distance_cm': int(m.group(4)),
    }


def select_bias_window(df):
    """Return the control-phase rows inside the bias-averaging window."""
    control = df[df['phase_type'] == 'control']
    if len(control) == 0:
        return control.iloc[0:0]
    t_end = control['time_seconds'].max()
    win_end = t_end - CONTROL_MARGIN_S
    win_start = win_end - CONTROL_WINDOW_S
    return control[(control['time_seconds'] >= win_start) &
                   (control['time_seconds'] <= win_end)]


def ir_bias_window_sd(df):
    """SD of calibrated IR distance inside the bias-averaging window."""
    window = select_bias_window(df)
    if len(window) < 2:
        return np.nan, len(window)
    values = window[IR_NOISE_COL].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, len(values)
    return float(np.std(values, ddof=1)), len(values)


def vl5_bias_window_sd(df):
    """SD of center-2x2 VL5 distance (cm), pooled across 4 cells and timestamps."""
    window = select_bias_window(df)
    if len(window) < 2:
        return np.nan, len(window)
    values = window[VL5_CENTER_CM_COLS].to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, len(values)
    return float(np.std(values, ddof=1)), len(values)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'control_noise')


def _distance_summary(results, sensor):
    rows = []
    for d in DISTANCES_CM:
        sub = results[(results['distance_cm'] == d) & results['sd_cm'].notna()]
        if len(sub) == 0:
            rows.append({'sensor': sensor, 'distance_cm': d, 'n_trials': 0,
                         'mean_sd_cm': np.nan, 'median_sd_cm': np.nan,
                         'std_of_sds_cm': np.nan,
                         'min_sd_cm': np.nan, 'max_sd_cm': np.nan})
            continue
        sds = sub['sd_cm'].to_numpy()
        rows.append({'sensor': sensor, 'distance_cm': d, 'n_trials': int(len(sds)),
                     'mean_sd_cm': round(float(np.mean(sds)), 4),
                     'median_sd_cm': round(float(np.median(sds)), 4),
                     'std_of_sds_cm': round(float(np.std(sds, ddof=1)) if len(sds) > 1 else 0.0, 4),
                     'min_sd_cm': round(float(sds.min()), 4),
                     'max_sd_cm': round(float(sds.max()), 4)})
    return pd.DataFrame(rows)


def analyze_sensor(sensor, sd_func):
    data_dir = IR_DIR if sensor == 'IR' else VL5_DIR
    csv_files = sorted(glob.glob(os.path.join(data_dir, f'{sensor}_*.csv')))
    if not csv_files:
        raise FileNotFoundError(f'No processed {sensor} CSVs found under {data_dir}')

    rows = []
    for path in csv_files:
        meta = parse_filename(path)
        if meta is None or meta['sensor'] != sensor:
            continue
        df = pd.read_csv(path)
        sd, n = sd_func(df)
        rows.append({'sensor': sensor, **meta,
                     'file': os.path.basename(path),
                     'sd_cm': round(sd, 4) if np.isfinite(sd) else np.nan,
                     'n_samples': n})

    trials = pd.DataFrame(rows)
    summary = _distance_summary(trials, sensor)
    return trials, summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ir_trials, ir_summary = analyze_sensor('IR', ir_bias_window_sd)
    vl5_trials, vl5_summary = analyze_sensor('VL5', vl5_bias_window_sd)

    trials = pd.concat([ir_trials, vl5_trials], ignore_index=True)
    summary = pd.concat([ir_summary, vl5_summary], ignore_index=True)

    trial_path = os.path.join(OUTPUT_DIR, 'trial_sds.csv')
    summary_path = os.path.join(OUTPUT_DIR, 'distance_summary.csv')
    trials.to_csv(trial_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f'Saved: {trial_path}')
    print(f'Saved: {summary_path}')


if __name__ == '__main__':
    main()
