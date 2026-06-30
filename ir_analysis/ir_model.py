"""Calibrated IR dust-response model for a Sharp PSD sensor (Beer-Lambert extinction + finite-kernel backscatter, PSD-coordinate centroid)."""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, '..', 'dust_model'))

from dust_model import (  # noqa: E402
    RHO_P,
    SETTLING_TIME_S,
    DT_S,
    get_measurement_concentration_timeseries,
)
from mie_properties import mie_efficiencies_bins  # noqa: E402

_trapz = getattr(np, 'trapezoid', None) or np.trapz


MEASUREMENT_DURATION_S = 900.0

ETA_BACKSCATTER = 0.00027464124

CONCENTRATION_SCALE = {0.2: 1.4956025, 0.8: 0.46882556}

X_MIN_M = 0.04
X_REG_M = 0.002
X_OVERLAP_M = 0.010046
P_OVERLAP = 2.0
N_QUAD = 240
SIGNAL_FLOOR = 1e-12


def _as_time_bin_array(C_bins_kg: np.ndarray) -> np.ndarray:
    C = np.asarray(C_bins_kg, dtype=float)
    if C.ndim == 1:
        C = C[np.newaxis, :]
    if C.ndim != 2:
        raise ValueError("C_bins_kg must be a 1D or 2D array.")
    if np.any(~np.isfinite(C)):
        raise ValueError("C_bins_kg contains non-finite values.")
    if np.any(C < 0.0):
        raise ValueError("C_bins_kg must be non-negative.")
    return C


def compute_extinction_coefficient(C_bins_kg: np.ndarray,
                                   d_rep_um: np.ndarray,
                                   Q_ext: np.ndarray,
                                   rho_p: float = RHO_P) -> np.ndarray:
    """Beer-Lambert extinction coefficient [1/m]: sum_i 3 Q_ext_i C_i / (2 rho_p d_i)."""
    C = _as_time_bin_array(C_bins_kg)
    d_m = np.asarray(d_rep_um, dtype=float) * 1.0e-6
    Q = np.asarray(Q_ext, dtype=float)
    if d_m.shape != Q.shape or C.shape[1] != len(d_m):
        raise ValueError("C_bins_kg, d_rep_um, and Q_ext dimensions do not match.")
    if np.any(d_m <= 0.0):
        raise ValueError("Representative diameters must be positive.")

    beta_bins = C * (3.0 * Q[None, :]) / (2.0 * rho_p * d_m[None, :])
    return beta_bins.sum(axis=1)


def compute_volume_backscatter_coefficient(C_bins_kg: np.ndarray,
                                           d_rep_um: np.ndarray,
                                           Q_back: np.ndarray,
                                           rho_p: float = RHO_P) -> np.ndarray:
    """Volume backscatter coefficient beta_R [1/(m sr)]: sum_i 3 Q_back_i C_i / (8 pi rho_p d_i)."""
    C = _as_time_bin_array(C_bins_kg)
    d_m = np.asarray(d_rep_um, dtype=float) * 1.0e-6
    Q = np.asarray(Q_back, dtype=float)
    if d_m.shape != Q.shape or C.shape[1] != len(d_m):
        raise ValueError("C_bins_kg, d_rep_um, and Q_back dimensions do not match.")
    if np.any(d_m <= 0.0):
        raise ValueError("Representative diameters must be positive.")

    beta_bins = C * (3.0 * Q[None, :]) / (8.0 * np.pi * rho_p * d_m[None, :])
    return beta_bins.sum(axis=1)


def dust_kernel(x_m: np.ndarray,
                x_reg_m: float = X_REG_M,
                x_overlap_m: float = X_OVERLAP_M,
                p_overlap: float = P_OVERLAP) -> np.ndarray:
    """Finite near-field backscatter collection kernel: overlap(x) / (x^2 + x_reg^2)."""
    x = np.asarray(x_m, dtype=float)
    x0 = max(float(x_reg_m), 0.0)
    if x_overlap_m <= 0.0:
        return np.ones_like(x) / (x * x + x0 * x0)
    overlap = 1.0 - np.exp(-np.maximum(x, 0.0) ** p_overlap /
                           float(x_overlap_m) ** p_overlap)
    return overlap / (x * x + x0 * x0)


def dust_biased_distance(
    d_true_m: float,
    alpha_ext: np.ndarray,
    beta_back: np.ndarray,
    eta: float = ETA_BACKSCATTER,
    x_min_m: float = X_MIN_M,
    x_reg_m: float = X_REG_M,
    x_overlap_m: float = X_OVERLAP_M,
    p_overlap: float = P_OVERLAP,
) -> np.ndarray:
    """Apparent distance [m] from a PSD-coordinate centroid model (q = 1/r)."""
    d = float(d_true_m)
    if d <= 0.0:
        raise ValueError("d_true_m must be positive.")
    alpha = np.atleast_1d(np.asarray(alpha_ext, dtype=float))
    beta = np.atleast_1d(np.asarray(beta_back, dtype=float))
    if alpha.shape != beta.shape:
        raise ValueError("alpha_ext and beta_back must have the same shape.")
    if np.any(alpha < 0.0) or np.any(beta < 0.0):
        raise ValueError("alpha_ext and beta_back must be non-negative.")

    S_T = np.exp(-2.0 * alpha * d)
    q_T = 1.0 / d

    if d <= x_min_m:
        return np.full_like(alpha, d)

    x = np.linspace(float(x_min_m), d, int(N_QUAD))
    Kx = dust_kernel(x, x_reg_m=x_reg_m,
                     x_overlap_m=x_overlap_m,
                     p_overlap=p_overlap)
    atten = np.exp(-2.0 * alpha[:, None] * x[None, :])
    dS_B_dx = eta * beta[:, None] * Kx[None, :] * atten

    S_B = _trapz(dS_B_dx, x, axis=1)
    q_B_int = _trapz(dS_B_dx * (1.0 / np.maximum(x, 1e-12))[None, :], x, axis=1)

    numerator = S_T * q_T + q_B_int
    denominator = S_T + S_B

    valid = denominator > SIGNAL_FLOOR
    q_pred = np.where(valid, numerator / np.maximum(denominator, SIGNAL_FLOOR), q_T)
    return 1.0 / np.maximum(q_pred, 1e-12)


def compute_measurement_optics(
    mass_g: float,
    t_meas_s: np.ndarray,
    settling_time_s: float = SETTLING_TIME_S,
    concentration_scale: float = 1.0,
) -> dict:
    """Compute dust concentrations and optical coefficients on measurement time."""
    if concentration_scale <= 0.0:
        raise ValueError("concentration_scale must be positive.")

    dust = get_measurement_concentration_timeseries(
        mass_g, t_meas_s, pre_measurement_s=settling_time_s)
    C_bins_kg = dust['C_bins_mg_m3'] * float(concentration_scale) * 1.0e-6
    d_rep_um = dust['d_rep_m'] * 1.0e6

    mie = mie_efficiencies_bins(d_rep_um)
    alpha_ext = compute_extinction_coefficient(C_bins_kg, d_rep_um, mie['Q_ext'])
    beta_back = compute_volume_backscatter_coefficient(C_bins_kg, d_rep_um, mie['Q_back'])

    return {
        't_s': np.asarray(t_meas_s, dtype=float),
        'alpha_ext_1_per_m': alpha_ext,
        'beta_back_1_per_m_sr': beta_back,
    }


def simulate_experiment(
    distance_cm: float,
    mass_g: float,
    settling_time_s: float = SETTLING_TIME_S,
    measurement_duration_s: float = MEASUREMENT_DURATION_S,
    eta: float = ETA_BACKSCATTER,
    x_min_m: float = X_MIN_M,
    x_reg_m: float = X_REG_M,
    x_overlap_m: float = X_OVERLAP_M,
    p_overlap: float = P_OVERLAP,
    concentration_scale: Optional[float] = None,
) -> dict:
    """Simulate IR sensor measurements; t_s is time since first sample (t_abs = settling_time_s + t_s)."""
    if concentration_scale is None:
        concentration_scale = CONCENTRATION_SCALE.get(float(mass_g), 1.0)

    t_meas = np.arange(0.0, float(measurement_duration_s) + DT_S, DT_S)
    optics = compute_measurement_optics(
        mass_g, t_meas, settling_time_s=settling_time_s,
        concentration_scale=concentration_scale)

    d_m = float(distance_cm) / 100.0
    alpha = optics['alpha_ext_1_per_m']
    beta = optics['beta_back_1_per_m_sr']

    d_est_m = dust_biased_distance(
        d_m, alpha, beta, eta=eta, x_min_m=x_min_m,
        x_reg_m=x_reg_m, x_overlap_m=x_overlap_m,
        p_overlap=p_overlap)

    return {
        't_s': t_meas,
        'alpha_ext_1_per_m': alpha,
        'T_roundtrip': np.exp(-2.0 * alpha * d_m),
        'distance_est_cm': d_est_m * 100.0,
    }
