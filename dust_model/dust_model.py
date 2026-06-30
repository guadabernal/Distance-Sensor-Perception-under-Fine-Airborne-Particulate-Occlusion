"""Size-resolved post-pulse dust decay in a sealed chamber (Lai-Nazaroff/NIST deposition, piecewise u*)."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

_trapz = getattr(np, 'trapezoid', None) or np.trapz


BOX_L = 0.69
BOX_W = 0.295
BOX_H = 0.26

V_BOX = BOX_L * BOX_W * BOX_H
A_FLOOR = BOX_L * BOX_W
A_CEILING = A_FLOOR
A_WALLS = 2.0 * (BOX_L * BOX_H + BOX_W * BOX_H)

RHO_P = 2650.0
RHO_AIR = 1.2
MU = 1.81e-5
G = 9.81
LAMBDA_MFP = 66e-9
CHI = 1.36
TEMP = 293.15
K_B = 1.380649e-23
NU_AIR = MU / RHO_AIR

CUNNINGHAM_A = 1.257
CUNNINGHAM_B = 0.42
CUNNINGHAM_C = 1.10

DT_S = 0.5
SETTLING_TIME_S = 150.0
T_TOTAL_S = 1200.0

ETA_REL = 0.80
T_MIX_S = 30.0
U_STAR_TURB = 0.05
U_STAR_QUIET = 0.003

ISO_12103_A0_DISTRIBUTION = {
    0.97: 11.8,
    1.38: 21.75,
    2.75: 59.85,
    5.50: 91.5,
    11.0: 99.1,
    22.0: 100.0,
}


def compute_size_bins() -> pd.DataFrame:
    """Size bins from the ISO A0 cumulative distribution."""
    bin_edges_um = [0.0] + sorted(ISO_12103_A0_DISTRIBUTION.keys())
    bins_data = []
    prev_cum_frac = 0.0

    for i in range(len(bin_edges_um) - 1):
        d_low_um = float(bin_edges_um[i])
        d_high_um = float(bin_edges_um[i + 1])
        cum_frac = ISO_12103_A0_DISTRIBUTION[d_high_um] / 100.0
        w_i = cum_frac - prev_cum_frac

        if d_low_um == 0.0:
            d_rep_um = 0.5 * d_high_um
        else:
            d_rep_um = math.sqrt(d_low_um * d_high_um)

        bins_data.append({
            'bin': i + 1,
            'd_low_um': d_low_um,
            'd_high_um': d_high_um,
            'd_rep_um': d_rep_um,
            'd_rep_m': d_rep_um * 1e-6,
            'w_i': w_i,
        })
        prev_cum_frac = cum_frac

    df_bins = pd.DataFrame(bins_data)
    df_bins['w_i'] = df_bins['w_i'] / df_bins['w_i'].sum()
    return df_bins


def cunningham_slip_correction(d_m: float) -> float:
    """Cunningham slip correction for particle diameter d_m [m]."""
    if d_m <= 0.0:
        raise ValueError("Particle diameter must be positive.")

    kn = 2.0 * LAMBDA_MFP / d_m
    return 1.0 + kn * (CUNNINGHAM_A + CUNNINGHAM_B * math.exp(-CUNNINGHAM_C / kn))


def stokes_settling_velocity(d_m: float) -> Tuple[float, float]:
    """Settling velocity [m/s] and slip correction [-] for diameter d_m [m]."""
    C_c = cunningham_slip_correction(d_m)
    v_s = ((RHO_P - RHO_AIR) * G * d_m ** 2 * C_c) / (18.0 * MU * CHI)
    return v_s, C_c


def particle_diffusivity(d_m: float) -> float:
    """Brownian diffusivity [m^2/s] for diameter d_m [m] (shape factor included)."""
    if d_m <= 0.0:
        raise ValueError("Particle diameter must be positive.")

    C_c = cunningham_slip_correction(d_m)
    return K_B * TEMP * C_c / (3.0 * math.pi * MU * CHI * d_m)


def compute_settling_velocities(df_bins: pd.DataFrame) -> pd.DataFrame:
    """Add slip, settling, and diffusivity diagnostics to the size-bin table."""
    df = df_bins.copy()

    C_c_list = []
    v_s_list = []
    D_p_list = []
    for _, row in df.iterrows():
        d_m = float(row['d_rep_m'])
        v_s, C_c = stokes_settling_velocity(d_m)
        D_p = particle_diffusivity(d_m)
        C_c_list.append(C_c)
        v_s_list.append(v_s)
        D_p_list.append(D_p)

    df['C_c'] = C_c_list
    df['v_s_m_s'] = v_s_list
    df['D_p_m2_s'] = D_p_list
    return df


def _eddy_diffusivity(y_plus: np.ndarray) -> np.ndarray:
    """Particle eddy diffusivity epsilon_p(y+) [m^2/s] (Lai-Nazaroff 3-layer fit)."""
    y_plus = np.asarray(y_plus, dtype=float)
    eps_over_nu = np.empty_like(y_plus)

    mask1 = y_plus <= 4.3
    mask2 = (y_plus > 4.3) & (y_plus <= 12.5)
    mask3 = y_plus > 12.5

    eps_over_nu[mask1] = 7.669e-4 * y_plus[mask1] ** 3.0
    eps_over_nu[mask2] = 1.0e-3 * y_plus[mask2] ** 2.8214
    eps_over_nu[mask3] = 1.07e-2 * y_plus[mask3] ** 1.8895

    return NU_AIR * eps_over_nu


def _lai_nazaroff_integral(d_m: float, u_star: float, n_points: int = 600) -> float:
    """Compute the Lai–Nazaroff integral I numerically."""
    if u_star <= 0.0:
        return math.inf

    r_m = 0.5 * d_m
    r_plus = r_m * u_star / NU_AIR

    if r_plus >= 30.0:
        return 1.0e-30

    y = np.linspace(max(r_plus, 0.0), 30.0, int(n_points))
    D_p = particle_diffusivity(d_m)
    eps = _eddy_diffusivity(y)
    integrand = NU_AIR / (eps + D_p)
    I = float(_trapz(integrand, y))
    return max(I, 1.0e-30)


def _surface_deposition_velocities(d_m: float, u_star: float) -> Tuple[float, float, float, float]:
    """Deposition velocities (wall, floor, ceiling) [m/s] and Lai-Nazaroff integral I."""
    v_s, _ = stokes_settling_velocity(d_m)

    if u_star <= 0.0:
        return 0.0, v_s, 0.0, math.inf

    I = _lai_nazaroff_integral(d_m, u_star)
    u_dv = u_star / I

    x = v_s * I / u_star
    if (v_s <= 0.0) or (abs(x) < 1.0e-12):
        u_du = u_dv
        u_dd = u_dv
    elif x > 50.0:
        u_du = v_s
        u_dd = 0.0
    else:
        u_du = v_s / (-math.expm1(-x))
        u_dd = v_s / math.expm1(x)

    return u_dv, u_du, u_dd, I


def _lai_nazaroff_loss_rate(d_m: float, u_star: float) -> float:
    """First-order chamber deposition loss coefficient kappa [1/s]."""
    u_dv, u_du, u_dd, _I = _surface_deposition_velocities(d_m, u_star)
    return (A_WALLS * u_dv + A_FLOOR * u_du + A_CEILING * u_dd) / V_BOX


def _add_deposition_diagnostics(df_bins: pd.DataFrame) -> pd.DataFrame:
    """Add surface-specific deposition diagnostics for the two u* stages."""
    df = df_bins.copy()
    cols: Dict[str, list] = {
        'u_dv_turb_m_s': [],
        'u_du_turb_m_s': [],
        'u_dd_turb_m_s': [],
        'kappa_turb_s_inv': [],
        'u_dv_quiet_m_s': [],
        'u_du_quiet_m_s': [],
        'u_dd_quiet_m_s': [],
        'kappa_quiet_s_inv': [],
    }

    for _, row in df.iterrows():
        d_m = float(row['d_rep_m'])

        u_dv_t, u_du_t, u_dd_t, _ = _surface_deposition_velocities(d_m, U_STAR_TURB)
        u_dv_q, u_du_q, u_dd_q, _ = _surface_deposition_velocities(d_m, U_STAR_QUIET)

        cols['u_dv_turb_m_s'].append(u_dv_t)
        cols['u_du_turb_m_s'].append(u_du_t)
        cols['u_dd_turb_m_s'].append(u_dd_t)
        cols['kappa_turb_s_inv'].append((A_WALLS * u_dv_t + A_FLOOR * u_du_t + A_CEILING * u_dd_t) / V_BOX)

        cols['u_dv_quiet_m_s'].append(u_dv_q)
        cols['u_du_quiet_m_s'].append(u_du_q)
        cols['u_dd_quiet_m_s'].append(u_dd_q)
        cols['kappa_quiet_s_inv'].append((A_WALLS * u_dv_q + A_FLOOR * u_du_q + A_CEILING * u_dd_q) / V_BOX)

    for key, values in cols.items():
        df[key] = values
    return df


def _validate_and_index_times(t_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted times and indices to restore the original order."""
    t = np.asarray(t_array, dtype=float)
    if t.ndim != 1:
        raise ValueError("t_array must be one-dimensional.")
    if np.any(~np.isfinite(t)):
        raise ValueError("t_array contains non-finite values.")
    if np.any(t < 0.0):
        raise ValueError("t_array must be non-negative; times are measured from model start.")

    sort_idx = np.argsort(t, kind='mergesort')
    t_sorted = t[sort_idx]
    return t_sorted, sort_idx


def _stage_overlap(t0: float, t1: float) -> Tuple[float, float]:
    """Overlap of [t0, t1] with turbulent and quiet stages."""
    if t1 < t0:
        raise ValueError("Time array must be non-decreasing after sorting.")

    turb_start = 0.0
    turb_end = T_MIX_S
    dt_turb = max(0.0, min(t1, turb_end) - max(t0, turb_start))
    dt_quiet = (t1 - t0) - dt_turb
    return dt_turb, dt_quiet


def compute_concentration(t_array: np.ndarray, df_bins: pd.DataFrame,
                          C_0: float) -> pd.DataFrame:
    """Per-bin and total concentrations [mg/m^3] at requested times [s] from t=0."""
    if C_0 < 0.0:
        raise ValueError("C_0 must be non-negative.")

    t_sorted, sort_idx = _validate_and_index_times(np.asarray(t_array, dtype=float))
    n_bins = len(df_bins)
    n_times = len(t_sorted)

    df_local = df_bins.copy()
    required_cols = {'kappa_turb_s_inv', 'kappa_quiet_s_inv'}
    if not required_cols.issubset(df_local.columns):
        df_local = _add_deposition_diagnostics(df_local)

    kappa_turb = df_local['kappa_turb_s_inv'].to_numpy(dtype=float)
    kappa_quiet = df_local['kappa_quiet_s_inv'].to_numpy(dtype=float)
    w_i = df_local['w_i'].to_numpy(dtype=float)

    C_init = w_i * float(C_0)
    C_sorted = np.zeros((n_times, n_bins), dtype=float)

    C_curr = C_init.copy()
    t_prev = 0.0
    for idx, t_now in enumerate(t_sorted):
        dt_turb, dt_quiet = _stage_overlap(t_prev, t_now)
        decay = np.exp(-(kappa_turb * dt_turb + kappa_quiet * dt_quiet))
        C_curr = C_curr * decay
        C_sorted[idx, :] = C_curr
        t_prev = t_now

    unsort_idx = np.empty_like(sort_idx)
    unsort_idx[sort_idx] = np.arange(n_times)
    C = C_sorted[unsort_idx, :]
    t_out = np.asarray(t_array, dtype=float)

    df_ts = pd.DataFrame({'time_s': t_out, 'time_min': t_out / 60.0})
    for i in range(n_bins):
        df_ts[f'C_{i + 1}_mg_m3'] = C[:, i]
    df_ts['C_total_mg_m3'] = np.sum(C, axis=1)
    return df_ts


def run_dust_model(mass_g: float) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run pipeline for mass_g [g]; returns (df_bins, df_ts, C_0)."""
    if mass_g < 0.0:
        raise ValueError("mass_g must be non-negative.")

    C_0 = ETA_REL * mass_g * 1000.0 / V_BOX
    df_bins = compute_size_bins()
    df_bins = compute_settling_velocities(df_bins)
    df_bins = _add_deposition_diagnostics(df_bins)
    t_array = np.arange(0.0, T_TOTAL_S + DT_S, DT_S)
    df_ts = compute_concentration(t_array, df_bins, C_0)
    return df_bins, df_ts, C_0


def get_concentration_timeseries(mass_g: float, t_array_s: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-bin and total concentrations at t_array_s [s] for mass_g [g]."""
    if mass_g < 0.0:
        raise ValueError("mass_g must be non-negative.")

    C_0 = ETA_REL * mass_g * 1000.0 / V_BOX
    df_bins = compute_size_bins()
    df_bins = compute_settling_velocities(df_bins)
    df_bins = _add_deposition_diagnostics(df_bins)
    df_ts = compute_concentration(t_array_s, df_bins, C_0)

    n_bins = len(df_bins)
    C_bins = np.column_stack([df_ts[f'C_{i+1}_mg_m3'].to_numpy(dtype=float)
                              for i in range(n_bins)])

    return {
        't_s': np.asarray(t_array_s, dtype=float),
        'C_bins_mg_m3': C_bins,
        'C_total_mg_m3': df_ts['C_total_mg_m3'].to_numpy(dtype=float),
        'd_rep_m': df_bins['d_rep_m'].to_numpy(dtype=float),
        'w_i': df_bins['w_i'].to_numpy(dtype=float),
        'C_0_mg_m3': C_0,
    }


def get_measurement_concentration_timeseries(
    mass_g: float,
    t_meas_s: np.ndarray,
    pre_measurement_s: float = SETTLING_TIME_S,
) -> Dict[str, np.ndarray]:
    """Concentrations on sensor timebase; samples at t_abs = pre_measurement_s + t_meas."""
    t_meas = np.asarray(t_meas_s, dtype=float)
    if t_meas.ndim != 1:
        raise ValueError("t_meas_s must be one-dimensional.")
    if np.any(~np.isfinite(t_meas)):
        raise ValueError("t_meas_s contains non-finite values.")
    if np.any(t_meas < 0.0):
        raise ValueError("Measurement times must be non-negative.")
    if pre_measurement_s < 0.0:
        raise ValueError("pre_measurement_s must be non-negative.")

    t_abs = float(pre_measurement_s) + t_meas
    out = get_concentration_timeseries(mass_g, t_abs)
    out['t_abs_s'] = t_abs
    out['t_meas_s'] = t_meas
    out['pre_measurement_s'] = float(pre_measurement_s)
    return out
