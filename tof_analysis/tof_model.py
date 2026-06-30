"""Semi-physical VL53L5CX ToF dust observation model: false-target evidence + wall-locked estimator bias."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional
import math, os, sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, '..', 'dust_model'))
import dust_model as aqm  # noqa: E402
from tof_mie_properties import mie_efficiencies_bins, WAVELENGTH_UM, M_COMPLEX  # noqa: E402

_trapz = getattr(np, 'trapezoid', None) or np.trapz

RHO_P = aqm.RHO_P
DT_S, SETTLING_TIME_S = aqm.DT_S, aqm.SETTLING_TIME_S
SIM_T_MAX_MIN, SIM_N_FRAMES, RANDOM_SEED = 15.0, 80, 42

D_REF, GAMMA, AMBIENT, R_OV = 0.30, 0.80, 0.02, 0.035
ALPHA_0_MEASURED, S_LIDAR = np.nan, 50.0
MEASUREMENT_DURATION_S = SIM_T_MAX_MIN * 60.0

@dataclass(frozen=True)
class ToFModelParams:
    concentration_scale_0p2: float = 0.1853149769
    concentration_scale_0p8: float = 0.2906416091
    backscatter_mode: str = 'lidar_ratio'
    s_eff_sr: float = S_LIDAR
    beta_mie_scale: float = 1.0
    dust_gain: float = 3.669467069e-05
    wall_signal_ref: float = 1.0
    wall_reflectance: float = GAMMA
    wall_distance_power: float = 2.0
    ambient_evidence: float = AMBIENT
    r_min_m: float = 0.035
    r_near_m: float = 0.220
    r_reg_m: float = 0.020
    r_overlap_m: float = R_OV
    overlap_power: float = 2.0
    n_quad: int = 80
    lock_intercept: float = 4.604121215
    lock_gain_evidence: float = 1.345771599
    ambient_lock_gain: float = 0.0
    target_order: str = 'strongest'
    estimator_window_sigma_m: float = 0.09731682767
    bias_scale: float = 3.682115749
    clean_bias_m: float = 0.0

    wall_bias_max_m: float = 0.02659078453
    wall_bias_centroid_gain: float = 0.02265304228
    wall_bias_size_power: float = 3.744355766
    wall_bias_driver_power: float = 2.069898363
    wall_bias_driver_ref_1_per_m: float = 1.0
    wall_bias_size_ref_um: float = 2.5
    wall_bias_distance_center_m: float = 0.2643392361
    wall_bias_distance_width_m: float = 0.07022328358
    sigma_turb: float = 0.2122254204
    wall_noise_base_m: float = 0.003
    wall_noise_signal_coeff_m: float = 0.002
    false_noise_m: float = 0.008
    invalid_probability_floor: float = 0.0
    window_deposition_velocity_m_s: float = 0.0
    window_loading_scale_kg_m2: float = 2.0e-5
    window_clean_rate_s_inv: float = 0.0
    window_wall_atten_coeff: float = 0.0
    xtalk_base: float = 0.0
    xtalk_gain: float = 0.0
    xtalk_range_m: float = 0.035

    false_target_gain: float = 4861.331651
    false_target_alpha_ref_1_per_m: float = 1.165534568
    false_target_alpha_power: float = 7.135206955
    false_target_range_crit_m: float = 0.3739893407
    false_target_range_width_m: float = 0.06944750729
    false_target_distance_ref_m: float = 0.40
    false_target_distance_power: float = -1.25
    false_target_decay_tau_ref_s: float = 15.56892092
    false_target_decay_distance_power: float = 1.348831349

    max_reported_range_m: float = 4.0

DEFAULT_PARAMS = ToFModelParams()
RNG = np.random.default_rng(RANDOM_SEED)

def concentration_scale_for_mass(mass_g, params=DEFAULT_PARAMS):
    m = float(mass_g)
    if abs(m - 0.2) < 1e-9: return float(params.concentration_scale_0p2)
    if abs(m - 0.8) < 1e-9: return float(params.concentration_scale_0p8)
    return 1.0

def _as2d(C):
    C = np.asarray(C, dtype=float)
    if C.ndim == 1: C = C[None, :]
    if C.ndim != 2 or np.any(~np.isfinite(C)) or np.any(C < 0):
        raise ValueError('concentration array must be finite, nonnegative, 1D/2D')
    return C

def compute_extinction_coefficient(C_bins_kg, d_rep_um, Q_ext, rho_p=RHO_P):
    C, d, Q = _as2d(C_bins_kg), np.asarray(d_rep_um)*1e-6, np.asarray(Q_ext)
    return np.sum(C * (3.0*Q[None, :])/(2.0*rho_p*d[None, :]), axis=1)

def compute_volume_backscatter_coefficient(C_bins_kg, d_rep_um, Q_back, rho_p=RHO_P):
    C, d, Q = _as2d(C_bins_kg), np.asarray(d_rep_um)*1e-6, np.asarray(Q_back)
    return np.sum(C * (3.0*Q[None, :])/(8.0*np.pi*rho_p*d[None, :]), axis=1)

def build_psd(mass_g=0.8):
    dust = aqm.get_measurement_concentration_timeseries(float(mass_g), np.array([0.0]))
    d_um = dust['d_rep_m'] * 1e6
    mie = mie_efficiencies_bins(d_um)
    return {'d_rep_um': d_um, 'd_rep_m': dust['d_rep_m'], 'w_i': dust['w_i'],
            'QEXT': mie['Q_ext'], 'QSCA': mie['Q_sca'], 'QBACK': mie['Q_back'],
            'QABS': mie['Q_abs'], 'OMEGA0': mie['omega_0'], 'G_ASYM': mie['g'],
            'AREA': np.pi*(dust['d_rep_m']/2)**2, 'D_UM': d_um, 'D_M': dust['d_rep_m'],
            'wavelength_um': WAVELENGTH_UM, 'm_complex': M_COMPLEX,
            'C0_mg': dust['C_0_mg_m3']}

def _window_loading(mass_g, t_meas, params):
    if params.window_deposition_velocity_m_s <= 0 and params.xtalk_gain == 0 and params.window_wall_atten_coeff == 0:
        return np.zeros_like(t_meas)
    t_abs = np.arange(0, SETTLING_TIME_S + float(np.max(t_meas)) + DT_S, DT_S)
    raw = aqm.get_concentration_timeseries(float(mass_g), t_abs)
    Ckg = raw['C_total_mg_m3'] * concentration_scale_for_mass(mass_g, params) * 1e-6
    q = np.zeros_like(t_abs)
    L0 = max(params.window_loading_scale_kg_m2, 1e-30)
    for i in range(1, len(t_abs)):
        dt = t_abs[i] - t_abs[i-1]
        source = max(params.window_deposition_velocity_m_s, 0.0)*Ckg[i-1]/L0
        q[i] = max(0.0, q[i-1] + dt*(source - max(params.window_clean_rate_s_inv, 0.0)*q[i-1]))
    return np.interp(SETTLING_TIME_S + t_meas, t_abs, q)

def _beta_effective(alpha, beta_mie, params):
    if params.backscatter_mode == 'lidar_ratio':
        return alpha / max(float(params.s_eff_sr), 1e-12)
    if params.backscatter_mode == 'mie':
        return beta_mie * float(params.beta_mie_scale)
    raise ValueError("backscatter_mode must be 'lidar_ratio' or 'mie'")

def build_dust_data(mass_g, t_meas_s=None, measurement_duration_s=MEASUREMENT_DURATION_S,
                    settling_time_s=SETTLING_TIME_S, params=DEFAULT_PARAMS):
    t = np.arange(0.0, float(measurement_duration_s) + DT_S, DT_S) if t_meas_s is None else np.asarray(t_meas_s, dtype=float)
    meas = aqm.get_measurement_concentration_timeseries(float(mass_g), t, pre_measurement_s=float(settling_time_s))
    scale = concentration_scale_for_mass(mass_g, params)
    C_ch = meas['C_bins_mg_m3']; C_loc = C_ch * scale; Ckg = C_loc * 1e-6
    psd = build_psd(mass_g)
    alpha_bins_ext = Ckg * (3.0 * psd['QEXT'][None, :]) / (2.0 * RHO_P * (psd['d_rep_um'][None, :] * 1.0e-6))
    alpha = alpha_bins_ext.sum(axis=1)
    alpha_s = compute_extinction_coefficient(Ckg, psd['d_rep_um'], psd['QSCA'])
    beta_mie = compute_volume_backscatter_coefficient(Ckg, psd['d_rep_um'], psd['QBACK'])
    size_weight = (psd['d_rep_um'] / max(float(params.wall_bias_size_ref_um), 1.0e-12)) ** float(params.wall_bias_size_power)
    wall_bias_driver = np.sum(alpha_bins_ext * size_weight[None, :], axis=1)
    beta = _beta_effective(alpha, beta_mie, params)
    sca_bins = Ckg * (3.0*psd['QSCA'][None, :])/(2.0*RHO_P*(psd['d_rep_um'][None, :]*1e-6))
    omega0 = np.divide(alpha_s, alpha, out=np.zeros_like(alpha), where=alpha>0)
    g_bulk = np.divide(np.sum(sca_bins*psd['G_ASYM'][None, :], axis=1), alpha_s, out=np.zeros_like(alpha_s), where=alpha_s>0)
    return {'t_s': t, 't_abs_s': meas['t_abs_s'], 'C_bins_chamber_mg_m3': C_ch,
            'C_total_chamber_mg_m3': C_ch.sum(axis=1), 'C_bins_mg_m3': C_loc,
            'C_total_mg_m3': C_loc.sum(axis=1), 'concentration_scale': scale,
            'q_window': _window_loading(mass_g, t, params), 'd_rep_m': meas['d_rep_m'],
            'w_i': meas['w_i'], 'C_0_mg_m3': meas['C_0_mg_m3'],
            'alpha_ext_1_per_m': alpha, 'alpha_sca_1_per_m': alpha_s,
            'beta_mie_1_per_m_sr': beta_mie, 'beta_back_1_per_m_sr': beta,
            'wall_bias_driver_1_per_m': wall_bias_driver,
            'omega0': omega0, 'g_bulk': g_bulk}

def _idx(t_s, t_arr):
    return int(np.clip(np.searchsorted(t_arr, float(t_s), side='right') - 1, 0, len(t_arr)-1))

def overlap_function(r, params=DEFAULT_PARAMS):
    r = np.asarray(r, dtype=float)
    return 1.0 - np.exp(-(np.maximum(r, 0.0)/max(params.r_overlap_m, 1e-12))**params.overlap_power)

def collection_kernel(r, params=DEFAULT_PARAMS):
    r = np.asarray(r, dtype=float)
    return overlap_function(r, params) / (r*r + max(params.r_reg_m,0.0)**2)

def wall_signal_clean(D, params=DEFAULT_PARAMS):
    return params.wall_signal_ref * (params.wall_reflectance/GAMMA) * (D_REF/max(float(D), 1e-9))**params.wall_distance_power

def dust_signal_density(r, alpha, beta_back, params=DEFAULT_PARAMS):
    r = np.asarray(r, dtype=float)
    return params.dust_gain * float(beta_back) * collection_kernel(r, params) * np.exp(-2.0*float(alpha)*r)

def _sigmoid_scalar(x):
    return float(1.0 / (1.0 + math.exp(-max(min(float(x), 60.0), -60.0))))

def dynamic_wall_bias(D_wall, alpha, E_wall, bias_driver=None, params=DEFAULT_PARAMS):
    """Wall-locked dust bias [m]: delta = K * G(D) * B(t,D) / E_wall (size-weighted pedestal driver)."""
    D = max(float(D_wall), 0.0)
    a = max(float(alpha), 0.0)
    Ew = max(float(E_wall), 1.0e-30)
    driver = a if bias_driver is None else max(float(bias_driver), 0.0)
    if D <= 0.0 or driver <= 0.0 or params.wall_bias_centroid_gain <= 0.0:
        return 0.0

    rmin = min(max(params.r_min_m, 1.0e-6), D)
    if D <= rmin:
        return 0.0

    ra = np.linspace(rmin, D, max(16, int(params.n_quad)))
    W = np.exp(-0.5 * ((ra - D) / max(params.estimator_window_sigma_m, 1.0e-9)) ** 2)
    K = collection_kernel(ra, params) * np.exp(-2.0 * a * ra)
    moment = float(_trapz((ra - D) * W * K, ra))

    width = max(float(params.wall_bias_distance_width_m), 1.0e-6)
    center = float(params.wall_bias_distance_center_m)
    range_gate = math.exp(-0.5 * ((D - center) / width) ** 2)

    driver_ref = max(float(params.wall_bias_driver_ref_1_per_m), 1.0e-30)
    driver_eff = (driver / driver_ref) ** max(float(params.wall_bias_driver_power), 0.0)

    bias = float(params.wall_bias_centroid_gain) * range_gate * driver_eff * moment / Ew
    return float(np.clip(bias, -max(float(params.wall_bias_max_m), 0.0), 0.0))

def transient_false_target_evidence(D_wall, alpha, E_dust_near, t_s=0.0, params=DEFAULT_PARAMS):
    """Transient near-field evidence for wall/false-lock competition (empirical, time-decaying)."""
    if params.false_target_gain <= 0.0 or E_dust_near <= 0.0:
        return 0.0
    D = max(float(D_wall), 0.0)
    a = max(float(alpha), 0.0)
    width = max(float(params.false_target_range_width_m), 1.0e-6)
    range_gate = _sigmoid_scalar((D - float(params.false_target_range_crit_m)) / width)
    alpha_ref = max(float(params.false_target_alpha_ref_1_per_m), 1.0e-12)
    alpha_term = (a / alpha_ref) ** max(float(params.false_target_alpha_power), 0.0)
    D_ref = max(float(params.false_target_distance_ref_m), 1.0e-6)
    distance_term = (D / D_ref) ** float(params.false_target_distance_power)
    tau = max(float(params.false_target_decay_tau_ref_s), 1.0e-6) * (D / D_ref) ** float(params.false_target_decay_distance_power)
    time_term = math.exp(-max(float(t_s), 0.0) / tau)
    return float(E_dust_near * params.false_target_gain * range_gate * alpha_term * distance_term * time_term)

def evidence_components(D_wall, alpha, beta_back, q_window=0.0, params=DEFAULT_PARAMS, t_s=0.0, bias_driver=None):
    D = float(D_wall); rmin = min(max(params.r_min_m, 1e-6), D)
    Ew = wall_signal_clean(D, params) * math.exp(-2.0*float(alpha)*D) * math.exp(-params.window_wall_atten_coeff*max(float(q_window),0.0))
    rhi = min(D, max(rmin, params.r_near_m))
    if rhi > rmin:
        rn = np.linspace(rmin, rhi, max(8, params.n_quad))
        dn = dust_signal_density(rn, alpha, beta_back, params)
        Edn = float(_trapz(dn, rn)); rf_d = float(_trapz(rn*dn, rn)/max(Edn,1e-30))
    else:
        Edn, rf_d = 0.0, rmin
    if D > rmin:
        ra = np.linspace(rmin, D, max(16, params.n_quad))
        da = dust_signal_density(ra, alpha, beta_back, params)
        W = np.exp(-0.5*((ra-D)/max(params.estimator_window_sigma_m,1e-9))**2)
        Edw = float(_trapz(W*da, ra)); bnum = float(_trapz((ra-D)*W*da, ra))
    else:
        Edw, bnum = 0.0, 0.0
    Ex = params.xtalk_base + params.xtalk_gain*max(float(q_window),0.0)
    Ef = transient_false_target_evidence(D, alpha, Edn, t_s=t_s, params=params)
    En = Edn + Ex + Ef
    rf = ((Edn + Ef)*rf_d + Ex*params.xtalk_range_m)/max(En,1e-30) if En>0 else rf_d
    legacy_wall_bias = params.bias_scale*bnum/max(Ew+Edw,1e-30)
    dynamic_bias = dynamic_wall_bias(D, alpha, Ew, bias_driver=bias_driver, params=params)
    wall_bias = legacy_wall_bias + dynamic_bias + params.clean_bias_m
    X = math.log((Ew+1e-30)/(En+1e-30)); Y = math.log((Ew+1e-30)/(params.ambient_evidence+1e-30))
    logit = params.lock_intercept + params.lock_gain_evidence*X + params.ambient_lock_gain*Y
    if params.target_order.lower() == 'closest':
        logit -= max(0.0, math.log((Edn + Ef + 1e-30)/(Ew+1e-30)))
    pwall = _sigmoid_scalar(logit)
    return {'E_wall':Ew, 'E_dust_near':Edn, 'E_false_target':Ef, 'E_dust_window':Edw, 'E_xtalk':Ex,
            'E_ambient':params.ambient_evidence, 'p_wall':pwall, 'wall_bias_m':wall_bias,
            'wall_bias_dynamic_m': dynamic_bias, 'wall_bias_reduced_m': dynamic_bias, 'wall_bias_legacy_m': legacy_wall_bias,
            'false_range_m':float(np.clip(rf, 0.0, D)), 'q_window':float(q_window), 'lock_logit':logit}

def deterministic_distance(D_wall, alpha, beta_back, q_window=0.0, params=DEFAULT_PARAMS, mode='most_likely', t_s=0.0, bias_driver=None):
    ev = evidence_components(D_wall, alpha, beta_back, q_window, params, t_s=t_s, bias_driver=bias_driver)
    wd = float(D_wall) + ev['wall_bias_m']; fd = ev['false_range_m']
    d = ev['p_wall']*wd + (1-ev['p_wall'])*fd if mode == 'expected' else (wd if ev['p_wall'] >= 0.5 else fd)
    return float(np.clip(d, 0.0, params.max_reported_range_m))

def measure_one_frame(D_wall, alpha, beta, omega0=None, g_bulk=None, q_window=0.0, add_noise=True,
                      rng=None, params=DEFAULT_PARAMS, return_details=False, t_s=0.0, bias_driver=None):
    rng = RNG if rng is None else rng
    a, b = float(alpha), float(beta)
    if add_noise and params.sigma_turb > 0:
        mult = math.exp(rng.normal(0, params.sigma_turb) - 0.5*params.sigma_turb**2); a *= mult; b *= mult
    ev = evidence_components(D_wall, a, b, q_window, params, t_s=t_s, bias_driver=bias_driver)
    if add_noise and rng.random() < params.invalid_probability_floor:
        val = np.nan
    elif (not add_noise) or rng.random() < ev['p_wall']:
        sig = (params.wall_noise_base_m + params.wall_noise_signal_coeff_m/math.sqrt(max(ev['E_wall'],1e-12))) if add_noise else 0
        val = float(D_wall) + ev['wall_bias_m'] + (rng.normal(0, sig) if add_noise else 0)
    else:
        val = ev['false_range_m'] + (rng.normal(0, params.false_noise_m) if add_noise else 0)
    val = float(np.clip(val, 0, params.max_reported_range_m)) if np.isfinite(val) else np.nan
    if np.isfinite(val): val = round(val*1000)/1000
    return (val, ev) if return_details else val

def time_series(D_wall, t_minutes, mass_g, n_frames=SIM_N_FRAMES, psd=None, dust_data=None, params=DEFAULT_PARAMS, seed=RANDOM_SEED):
    dust_data = build_dust_data(mass_g, params=params) if dust_data is None else dust_data
    rng = np.random.default_rng(seed); t_minutes = np.asarray(t_minutes, dtype=float)
    out = np.full((len(t_minutes), int(n_frames)), np.nan)
    bias_driver_arr = dust_data.get('wall_bias_driver_1_per_m', dust_data['alpha_ext_1_per_m'])
    for i, tm in enumerate(t_minutes):
        j = _idx(tm*60, dust_data['t_s'])
        for k in range(int(n_frames)):
            out[i,k] = measure_one_frame(D_wall, dust_data['alpha_ext_1_per_m'][j], dust_data['beta_back_1_per_m_sr'][j],
                                         dust_data['omega0'][j], dust_data['g_bulk'][j], dust_data['q_window'][j], True, rng, params, t_s=tm*60.0,
                                         bias_driver=bias_driver_arr[j])
    return out

def noiseless_series(D_wall, t_minutes, mass_g, psd=None, dust_data=None, params=DEFAULT_PARAMS, mode='most_likely'):
    dust_data = build_dust_data(mass_g, params=params) if dust_data is None else dust_data
    bias_driver_arr = dust_data.get('wall_bias_driver_1_per_m', dust_data['alpha_ext_1_per_m'])
    vals = []
    for tm in np.asarray(t_minutes, dtype=float):
        j = _idx(tm*60, dust_data['t_s'])
        vals.append(deterministic_distance(D_wall, dust_data['alpha_ext_1_per_m'][j],
                                           dust_data['beta_back_1_per_m_sr'][j],
                                           dust_data['q_window'][j], params, mode, t_s=tm*60.0,
                                           bias_driver=bias_driver_arr[j]))
    return np.asarray(vals)

def expected_series(D_wall, t_minutes, mass_g, psd=None, dust_data=None, params=DEFAULT_PARAMS):
    return noiseless_series(D_wall, t_minutes, mass_g, psd, dust_data, params, mode='expected')

def evidence_series(D_wall, t_minutes, mass_g, psd=None, dust_data=None, params=DEFAULT_PARAMS):
    dust_data = build_dust_data(mass_g, params=params) if dust_data is None else dust_data
    rows = {}; t_minutes = np.asarray(t_minutes, dtype=float)
    bias_driver_arr = dust_data.get('wall_bias_driver_1_per_m', dust_data['alpha_ext_1_per_m'])
    for tm in t_minutes:
        i = _idx(tm*60, dust_data['t_s'])
        ev = evidence_components(D_wall, dust_data['alpha_ext_1_per_m'][i], dust_data['beta_back_1_per_m_sr'][i], dust_data['q_window'][i], params, t_s=tm*60.0,
                                 bias_driver=bias_driver_arr[i])
        for k,v in ev.items(): rows.setdefault(k, []).append(v)
    out = {k:np.asarray(v,float) for k,v in rows.items()}; out['t_min']=t_minutes; out['t_s']=t_minutes*60
    return out

def compute_evidence_series(D_wall, dust_data, mass_g=0.8, psd=None, params=DEFAULT_PARAMS):
    return evidence_series(D_wall, np.asarray(dust_data['t_s'])/60.0, mass_g, psd, dust_data, params)

def wall_locked_distance(D_wall, alpha_ext, beta_back, E_wall=None, params=DEFAULT_PARAMS, q_window=None, bias_driver=None):
    a = np.atleast_1d(np.asarray(alpha_ext,float)); b = np.atleast_1d(np.asarray(beta_back,float))
    q = np.zeros_like(a) if q_window is None else np.atleast_1d(np.asarray(q_window,float))
    bd = a if bias_driver is None else np.atleast_1d(np.asarray(bias_driver,float))
    return np.array([float(D_wall)+evidence_components(D_wall, ai, bi, qi, params, bias_driver=bdi)['wall_bias_m'] for ai,bi,qi,bdi in zip(a,b,q,bd)])

def compute_snap_threshold(D_wall, mass_g, psd=None, dust_data=None, params=DEFAULT_PARAMS):
    T = np.linspace(0, SIM_T_MAX_MIN, 90); ev = evidence_series(D_wall, T, mass_g, psd, dust_data, params)
    good = np.where(ev['p_wall'] >= 0.5)[0]
    if len(good) == 0: return None, None, None
    dust_data = build_dust_data(mass_g, params=params) if dust_data is None else dust_data
    a = np.interp(T[good[0]]*60, dust_data['t_s'], dust_data['alpha_ext_1_per_m'])
    return float(T[good[0]]), float(a), float(a)

def simulate_observation(D_wall_m, mass_g, t_meas_s=None, params=DEFAULT_PARAMS, n_frames=SIM_N_FRAMES, stochastic=False, seed=RANDOM_SEED):
    t = np.arange(0, MEASUREMENT_DURATION_S + DT_S, DT_S) if t_meas_s is None else np.asarray(t_meas_s,float)
    dust = build_dust_data(mass_g, t_meas_s=t, params=params)
    if stochastic:
        d = time_series(D_wall_m, t/60, mass_g, n_frames=n_frames, dust_data=dust, params=params, seed=seed)
        return {**dust, 'distances_m': d, 'distances_cm': d*100}
    d = expected_series(D_wall_m, t/60, mass_g, dust_data=dust, params=params)
    return {**dust, 'distance_m': d, 'distance_cm': d*100}

def simulate_experiment(distance_cm, mass_g, params=DEFAULT_PARAMS, measurement_duration_s=MEASUREMENT_DURATION_S):
    t = np.arange(0, measurement_duration_s + DT_S, DT_S); D = float(distance_cm)/100
    sim = simulate_observation(D, mass_g, t, params=params, stochastic=False)
    ev = compute_evidence_series(D, sim, mass_g, params=params)
    return {**sim, 'distance_true_cm': np.full_like(t, float(distance_cm)), 'distance_est_cm': sim['distance_cm'],
            'distance_wall_cm': wall_locked_distance(D, sim['alpha_ext_1_per_m'], sim['beta_back_1_per_m_sr'], params=params, q_window=sim['q_window'], bias_driver=sim.get('wall_bias_driver_1_per_m'))*100,
            'p_wall': ev['p_wall'], 'f_false_lock_expected': 1-ev['p_wall'], 'T_roundtrip': np.exp(-2*sim['alpha_ext_1_per_m']*D),
            'beta_ext_1_per_m': sim['alpha_ext_1_per_m'], 'beta_back_1_per_m': sim['beta_back_1_per_m_sr']}

def print_model_summary(mass_g, params=DEFAULT_PARAMS):
    dust = build_dust_data(mass_g, params=params)
    print('='*68); print('  Corrected VL53L5CX dust-observation model'); print('='*68)
    print(f'  loading={mass_g:g} g, measurement start={SETTLING_TIME_S:.0f} s')
    print(f'  C_chamber(t=0 meas)={dust["C_total_chamber_mg_m3"][0]:.2f} mg/m^3')
    print(f'  alpha_ext={dust["alpha_ext_1_per_m"][0]:.5g} 1/m, beta_eff={dust["beta_back_1_per_m_sr"][0]:.5g} 1/(m sr)')
    print(f'  beta_mie={dust["beta_mie_1_per_m_sr"][0]:.5g} 1/(m sr), mode={params.backscatter_mode}')
    for D in [0.1,0.2,0.3,0.4,0.5,0.6]:
        ev = evidence_components(D, dust['alpha_ext_1_per_m'][0], dust['beta_back_1_per_m_sr'][0], dust['q_window'][0], params, t_s=0.0)
        print(f'  D={D*100:2.0f} cm p_wall={ev["p_wall"]:.3f}, bias={ev["wall_bias_m"]*100:.3f} cm, false={ev["false_range_m"]*100:.1f} cm, E_false={ev["E_false_target"]:.3g}, bias_dyn={ev.get("wall_bias_dynamic_m", ev["wall_bias_m"])*100:.3f} cm')

__all__ = [name for name in globals() if not name.startswith('_')]
if __name__ == '__main__':
    print_model_summary(0.8); print(); print_model_summary(0.2)
