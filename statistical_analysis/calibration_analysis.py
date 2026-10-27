"""White-paper LUT calibration fits for both sensors, with in-sample and leave-one-distance-out residuals."""

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'data', 'calibration')
IR_LUT_CSV = os.path.join(CALIBRATION_DIR, 'ir_white_paper_lut.csv')
VL5_LUT_CSV = os.path.join(CALIBRATION_DIR, 'vl53l5cx_white_paper_lut.csv')


def ir_model(voltage, A, B):
    """d(V) = (A / V)^(1/B), NaN for non-positive voltage."""
    voltage = np.asarray(voltage, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.power(A / voltage, 1.0 / B)
        return np.where(voltage > 0, d, np.nan)


def fit_ir(voltages, distances):
    """Nonlinear least squares with true distance as the dependent quantity."""
    popt, _ = curve_fit(ir_model, np.asarray(voltages, dtype=float),
                        np.asarray(distances, dtype=float),
                        p0=[18.0, 0.85], maxfev=20000)
    return float(popt[0]), float(popt[1])


def vl5_inverse(measured_cm, a, b):
    """D_calibrated = (D_measured - b) / a."""
    return (np.asarray(measured_cm, dtype=float) - b) / a


def fit_vl5(true_cm, measured_cm):
    """Ordinary least-squares line: measured = a * true + b."""
    a, b = np.polyfit(np.asarray(true_cm, dtype=float),
                      np.asarray(measured_cm, dtype=float), 1)
    return float(a), float(b)


def vl5_calibration_cm(measured_mm, fits):
    """LUT-fit calibration of raw VL53L5CX zone readings [mm] to cm."""
    return vl5_inverse(np.asarray(measured_mm, dtype=float) / 10.0,
                       fits['vl5_slope'], fits['vl5_intercept_cm'])


def load_luts():
    return pd.read_csv(IR_LUT_CSV), pd.read_csv(VL5_LUT_CSV)


def compute_fits(ir_lut, vl5_lut):
    """IR fit on median voltages (primary) and ghost-cleaned means; VL5 linear fit."""
    A_med, B_med = fit_ir(ir_lut['voltage_median_v'], ir_lut['distance_cm'])
    A_cln, B_cln = fit_ir(ir_lut['voltage_clean_mean_v'], ir_lut['distance_cm'])
    a, b = fit_vl5(vl5_lut['distance_true_cm'], vl5_lut['distance_measured_mean_cm'])
    return {'ir_A': A_med, 'ir_B': B_med,
            'ir_clean_mean_A': A_cln, 'ir_clean_mean_B': B_cln,
            'vl5_slope': a, 'vl5_intercept_cm': b}


def _error_stats(residuals):
    r = np.asarray(residuals, dtype=float)
    return float(np.sqrt(np.mean(r ** 2))), float(np.mean(np.abs(r))), float(np.max(np.abs(r)))


def _fit_rows(sensor, basis, d_true, inputs, pred, loo, names, params):
    """One summary row plus per-point residual rows for one fit."""
    residual_rows = []
    for is_loo, p in ((False, pred), (True, loo)):
        for dt, xi, pi in zip(d_true, inputs, p):
            residual_rows.append({
                'sensor_code': sensor, 'fit_basis': basis,
                'distance_true_cm': dt, 'input_measurement': xi,
                'distance_predicted_cm': pi, 'residual_cm': pi - dt,
                'abs_residual_cm': abs(pi - dt), 'is_leave_one_out': is_loo,
            })
    rmse, mae, max_abs = _error_stats(pred - d_true)
    loo_rmse, loo_mae, _ = _error_stats(loo - d_true)
    summary = {
        'sensor_code': sensor, 'fit_basis': basis,
        'parameter_1_name': names[0], 'parameter_1_value': params[0],
        'parameter_2_name': names[1], 'parameter_2_value': params[1],
        'n_calibration_distances': len(d_true),
        'residual_df': len(d_true) - 2,
        'in_sample_rmse_cm': rmse, 'in_sample_mae_cm': mae,
        'max_abs_error_cm': max_abs,
        'loocv_rmse_cm': loo_rmse, 'loocv_mae_cm': loo_mae,
        'calibration_range_min_cm': float(d_true.min()),
        'calibration_range_max_cm': float(d_true.max()),
    }
    return summary, residual_rows


def build_calibration_tables(ir_lut, vl5_lut, fits):
    """Return (calibration_fit_summary, calibration_residuals)."""
    summary_rows, residual_rows = [], []

    d_true = ir_lut['distance_cm'].values.astype(float)
    for basis, col, A, B in (
            ('median', 'voltage_median_v', fits['ir_A'], fits['ir_B']),
            ('clean_mean', 'voltage_clean_mean_v',
             fits['ir_clean_mean_A'], fits['ir_clean_mean_B'])):
        v = ir_lut[col].values.astype(float)
        loo = np.empty_like(d_true)
        for i in range(len(d_true)):
            mask = np.arange(len(d_true)) != i
            loo[i] = ir_model(v[i], *fit_ir(v[mask], d_true[mask]))
        summary, rows = _fit_rows('IR', basis, d_true, v, ir_model(v, A, B), loo,
                                  ('A', 'B'), (A, B))
        summary_rows.append(summary)
        residual_rows += rows

    dt = vl5_lut['distance_true_cm'].values.astype(float)
    dm = vl5_lut['distance_measured_mean_cm'].values.astype(float)
    a, b = fits['vl5_slope'], fits['vl5_intercept_cm']
    loo = np.empty_like(dt)
    for i in range(len(dt)):
        mask = np.arange(len(dt)) != i
        loo[i] = vl5_inverse(dm[i], *fit_vl5(dt[mask], dm[mask]))
    summary, rows = _fit_rows('VL5', 'mean_measured', dt, dm, vl5_inverse(dm, a, b), loo,
                              ('a', 'b_cm'), (a, b))
    summary_rows.append(summary)
    residual_rows += rows

    residuals = pd.DataFrame(residual_rows).sort_values(
        ['sensor_code', 'fit_basis', 'is_leave_one_out', 'distance_true_cm'],
        kind='stable').reset_index(drop=True)
    return pd.DataFrame(summary_rows), residuals
