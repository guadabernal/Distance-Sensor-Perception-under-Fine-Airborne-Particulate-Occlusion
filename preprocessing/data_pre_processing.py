"""Process raw IR/VL5 sensor CSVs into per-experiment processed CSVs with calibrated distances and bias correction."""

import os
import glob
import re
import numpy as np
import pandas as pd
from collections import Counter
from scipy.signal import savgol_filter


IR_CAL_A = 18.125
IR_CAL_B = 0.844
VL5_CAL_OFFSET = 0.441
VL5_CAL_SCALE = 1.02


CONTROL_WINDOW_S = 20.0
CONTROL_MARGIN_S = 5.0
VL5_CENTER_2X2 = [27, 28, 35, 36]
VL5_FALSE_LOCK_FRAC = 0.70


def detect_bad_start_rows(timestamps, threshold_ratio=10):
    """Skip initial rows with abnormally large timestamp gaps (sensor warmup)."""
    if len(timestamps) < 2:
        return 0
    diffs = np.diff(timestamps[:min(100, len(timestamps))])
    valid_diffs = diffs[diffs > 0]
    if len(valid_diffs) == 0:
        return 0
    median_diff = np.median(valid_diffs)
    start_idx = 0
    for i, diff in enumerate(diffs):
        if diff > threshold_ratio * median_diff:
            start_idx = i + 1
    return start_idx


def calibrate_ir_distance(voltage):
    voltage = np.asarray(voltage, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        distance = np.power(IR_CAL_A / voltage, 1.0 / IR_CAL_B)
        distance = np.where(voltage > 0, distance, np.nan)
    return distance


def calibrate_vl5_distance(measured_mm):
    measured_mm = np.asarray(measured_mm, dtype=float)
    return ((measured_mm + VL5_CAL_OFFSET) / VL5_CAL_SCALE) / 10.0


def compute_ir_filtered_voltage(voltage_series, window=5):
    """Centered rolling median (width 5) — immune to isolated S/H ghost spikes."""
    filtered = voltage_series.rolling(
        window=window, center=True, min_periods=3
    ).median()
    filtered = filtered.bfill().ffill()
    return filtered


def flag_ir_ghost_samples(voltage_series, voltage_filtered, ghost_ratio=4.0):
    """Flag samples where raw V deviates above filtered V (informational; not used to exclude)."""
    residuals = voltage_series - voltage_filtered

    rolling_mad = residuals.abs().rolling(
        window=15, center=True, min_periods=5
    ).median()

    MAD_FLOOR = 0.003
    rolling_mad = rolling_mad.clip(lower=MAD_FLOOR)

    is_ghost = residuals > (ghost_ratio * rolling_mad)
    is_ghost = is_ghost.fillna(False).astype(bool)
    return is_ghost


def smooth_ir_voltage(voltage_filtered, window_length=21, polyorder=3):
    """Savitzky-Golay smoothing on already-median-filtered IR voltage."""
    values = np.asarray(voltage_filtered, dtype=float)

    effective_window = window_length
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    if len(values) < effective_window:
        effective_window = max(min_window, len(values))
        if effective_window % 2 == 0:
            effective_window -= 1
    if len(values) < min_window:
        return values.copy()

    return savgol_filter(values, effective_window, polyorder)


def compute_control_bias_offset(df, true_distance_cm, sensor_type):
    """Compute offset using a stable window inside the control phase."""
    control = df[df['phase_type'] == 'control']
    if len(control) == 0:
        return 0.0

    t_end = control['time_seconds'].max()
    win_end = t_end - CONTROL_MARGIN_S
    win_start = win_end - CONTROL_WINDOW_S
    tail = control[(control['time_seconds'] >= win_start) &
                   (control['time_seconds'] <= win_end)]
    if len(tail) == 0:
        tail = control

    if sensor_type == 'VL5':
        center_cal_cols = [f'cell_{i}_cm' for i in VL5_CENTER_2X2]
        mean_control = tail[center_cal_cols].values.mean()
    elif 'distance_cm_filtered' in tail.columns:
        mean_control = tail['distance_cm_filtered'].median()
    else:
        mean_control = tail['distance_cm_calibrated'].median()

    return true_distance_cm - mean_control


def load_ir_data(filepath):
    df = pd.read_csv(filepath, header=None, names=['timestamp', 'raw_adc', 'voltage', 'phase'])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    start_idx = detect_bad_start_rows(df['timestamp'].values)
    df = df.iloc[start_idx:].reset_index(drop=True)
    return df[['timestamp', 'voltage', 'phase']].rename(columns={'voltage': 'value'})


def load_vl5_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    col_counts = [len(line.strip().split(',')) for line in lines]
    most_common_cols = Counter(col_counts).most_common(1)[0][0]
    valid_lines = [line for line in lines if len(line.strip().split(',')) == most_common_cols]

    cell_cols = [f'cell_{i}' for i in range(64)]
    data = []
    for line in valid_lines:
        parts = line.strip().split(',')
        try:
            timestamp = float(parts[0])
            all_values = [float(x) for x in parts[2:-1]]
            phase = int(parts[-1])
            if len(all_values) != 64:
                continue
            data.append([timestamp, phase] + all_values)
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(data, columns=['timestamp', 'phase'] + cell_cols)
    start_idx = detect_bad_start_rows(df['timestamp'].values)
    df = df.iloc[start_idx:].reset_index(drop=True)
    return df


def filter_control_and_dust_phases(df):
    max_phase = int(df['phase'].max())
    dust_phase = max_phase
    control_phase = 2 if max_phase == 7 else 1

    control_rows = df[df['phase'] == control_phase]
    dust_rows = df[df['phase'] == dust_phase]

    if max_phase == 7 and len(control_rows) > 0:
        last_ts = control_rows['timestamp'].iloc[-1]
        control_rows = control_rows[control_rows['timestamp'] >= last_ts - 30000]

    filtered = pd.concat([control_rows, dust_rows]).sort_index().copy()
    filtered['phase_type'] = filtered['phase'].apply(
        lambda p: 'control' if p == control_phase else 'measurement'
    )
    return filtered


def parse_filename(filename):
    match = re.search(r'(\d{8})_(\d{6})_EXP(\d+)_(\d+\.?\d*)g_([A-Za-z0-9]+)_(\d+)', filename)
    if match:
        return {
            'trial': int(match.group(3)),
            'grams': float(match.group(4)),
            'sensor': match.group(5),
            'distance_cm': int(match.group(6)),
        }
    return None


_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data', 'raw_sensor_logs')
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, 'data', 'processed_experiments')


def process_sensor_data():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    folder_configs = {
        'IR2gData': ('IR', 0.2, load_ir_data),
        'IR8gData': ('IR', 0.8, load_ir_data),
        'VL52gData': ('VL5', 0.2, load_vl5_data),
        'VL58gData': ('VL5', 0.8, load_vl5_data),
    }

    for folder, (sensor_type, dust_grams, loader) in folder_configs.items():
        folder_path = os.path.join(RAW_DATA_DIR, folder)
        sensor_output_dir = os.path.join(PROCESSED_DIR, sensor_type)
        os.makedirs(sensor_output_dir, exist_ok=True)

        csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
        print(f"Processing {folder} ({len(csv_files)} files)...")

        for filepath in csv_files:
            filename = os.path.basename(filepath)
            metadata = parse_filename(filename)
            trial = metadata['trial']
            distance_cm = metadata['distance_cm']

            df = loader(filepath)
            df = filter_control_and_dust_phases(df)

            parts = []
            for pt in df['phase_type'].unique():
                sub = df[df['phase_type'] == pt].copy()
                t0 = sub['timestamp'].iloc[0]
                sub['time_normalized'] = sub['timestamp'] - t0
                sub['time_seconds'] = sub['time_normalized'] / 1000.0
                parts.append(sub)
            df = pd.concat(parts, ignore_index=True)

            if sensor_type == 'IR':
                df = df.rename(columns={'value': 'voltage'})
                df['distance_cm_calibrated'] = calibrate_ir_distance(df['voltage'].values)

                filtered_parts = []
                ghost_parts = []
                for pt in ['control', 'measurement']:
                    phase_idx = df.index[df['phase_type'] == pt]
                    if len(phase_idx) == 0:
                        continue
                    phase_voltage = df.loc[phase_idx, 'voltage']
                    filt = compute_ir_filtered_voltage(phase_voltage, window=5)
                    ghost = flag_ir_ghost_samples(phase_voltage, filt)
                    filtered_parts.append(filt)
                    ghost_parts.append(ghost)

                df['voltage_filtered'] = pd.concat(filtered_parts)
                df['is_ghost'] = pd.concat(ghost_parts)
                df['distance_cm_filtered'] = calibrate_ir_distance(df['voltage_filtered'].values)

                smooth_parts = []
                for pt in ['control', 'measurement']:
                    phase_idx = df.index[df['phase_type'] == pt]
                    if len(phase_idx) == 0:
                        continue
                    phase_filt = df.loc[phase_idx, 'voltage_filtered']
                    smoothed = smooth_ir_voltage(phase_filt)
                    smooth_parts.append(pd.Series(smoothed, index=phase_idx))

                df['voltage_smooth'] = pd.concat(smooth_parts)
                df['distance_cm_smooth'] = calibrate_ir_distance(df['voltage_smooth'].values)

                cols = ['time_seconds', 'timestamp',
                        'voltage', 'voltage_filtered', 'voltage_smooth',
                        'distance_cm_calibrated', 'distance_cm_filtered', 'distance_cm_smooth',
                        'phase_type', 'is_ghost']
            elif sensor_type == 'VL5':
                cell_cols = [f'cell_{i}' for i in range(64)]
                cal_cols = [f'cell_{i}_cm' for i in range(64)]
                for raw_col, cal_col in zip(cell_cols, cal_cols):
                    df[cal_col] = calibrate_vl5_distance(df[raw_col].values)
                fl_cols = [f'cell_{ci}_fl' for ci in VL5_CENTER_2X2]
                cols = (['time_seconds', 'timestamp', 'phase_type']
                        + cell_cols + cal_cols + fl_cols)

            offset = compute_control_bias_offset(df, distance_cm, sensor_type)
            if sensor_type == 'VL5':
                for cal_col in cal_cols:
                    df[cal_col] += offset
            else:
                df['distance_cm_calibrated'] += offset
                if 'distance_cm_filtered' in df.columns:
                    df['distance_cm_filtered'] += offset
                if 'distance_cm_smooth' in df.columns:
                    df['distance_cm_smooth'] += offset

            if sensor_type == 'VL5':
                threshold = VL5_FALSE_LOCK_FRAC * distance_cm
                for ci in VL5_CENTER_2X2:
                    cm_col = f'cell_{ci}_cm'
                    fl_col = f'cell_{ci}_fl'
                    df[fl_col] = (df[cm_col] < threshold) | df[cm_col].isna()
                    df.loc[df['phase_type'] == 'control', fl_col] = False

            df['bias_offset_cm'] = offset
            df = df[cols + ['bias_offset_cm']]

            output_filename = f"{sensor_type}_{dust_grams}g_TRIAL{trial}_{distance_cm}cm.csv"
            df.to_csv(os.path.join(sensor_output_dir, output_filename), index=False)
            if sensor_type == 'IR' and 'is_ghost' in df.columns:
                n_ghost = df['is_ghost'].sum()
                pct_ghost = 100 * n_ghost / len(df) if len(df) > 0 else 0
                print(f"  {filename} -> {sensor_type}/{output_filename} "
                      f"({len(df)} rows, offset {offset:+.3f} cm, "
                      f"ghost: {n_ghost}/{len(df)} = {pct_ghost:.1f}%)")
            elif sensor_type == 'VL5':
                meas = df[df['phase_type'] == 'measurement']
                fl_any = meas[fl_cols].any(axis=1)
                n_fl = fl_any.sum() if len(meas) > 0 else 0
                pct_fl = 100 * n_fl / len(meas) if len(meas) > 0 else 0
                print(f"  {filename} -> {sensor_type}/{output_filename} "
                      f"({len(df)} rows, offset {offset:+.3f} cm, "
                      f"false-lock: {n_fl}/{len(meas)} = {pct_fl:.1f}%)")
            else:
                print(f"  {filename} -> {sensor_type}/{output_filename} ({len(df)} rows, offset {offset:+.3f} cm)")

    print("Done!")


if __name__ == '__main__':
    process_sensor_data()
