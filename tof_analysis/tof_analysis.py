"""VL53L5CX ToF model vs experiment pipeline; outputs timeseries_metrics.csv and scalar_metrics.csv."""

import os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as _wasserstein

from tof_model import (
    time_series, build_psd, build_dust_data,
    SIM_N_FRAMES, SETTLING_TIME_S, DEFAULT_PARAMS, ToFModelParams,
)


CENTER_CELLS = [27, 28, 35, 36]
CENTER_CM_COLS = [f'cell_{i}_cm' for i in CENTER_CELLS]
CENTER_FL_COLS = [f'cell_{i}_fl' for i in CENTER_CELLS]

DISTANCES_CM = [10, 20, 30, 40, 50, 60]
LOADINGS_G = [0.2, 0.8]
TRIALS = [1, 2, 3]

TIME_BIN_WIDTH_S = 30.0
STEADY_STATE_START_S = 600.0

FL_THRESHOLD_FRAC = 0.70
N_FRAMES_ANALYSIS = 80

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data', 'processed_experiments', 'VL5')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics')


def phase_time_seconds(df_phase):
    """Phase-local time [s], rebased to first sample (robust to global/per-phase convention)."""
    t = np.asarray(df_phase['time_seconds'].values, dtype=float)
    if len(t) == 0:
        return t
    return t - t[0]


def load_experiment(grams, trial, dist_cm):
    """Measurement-phase DataFrame with center 2x2 cells melted to one row per (timestamp, cell)."""
    filename = f"VL5_{grams}g_TRIAL{trial}_{dist_cm}cm.csv"
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df = df[df['phase_type'] == 'measurement'].copy()
    if len(df) == 0:
        return None

    available_fl = [c for c in CENTER_FL_COLS if c in df.columns]
    has_fl = len(available_fl) == len(CENTER_FL_COLS)

    melted_parts = []
    for ci in CENTER_CELLS:
        cm_col = f'cell_{ci}_cm'
        fl_col = f'cell_{ci}_fl'
        part = pd.DataFrame({'time_seconds': phase_time_seconds(df)})
        part['distance_cm'] = df[cm_col].values
        part['is_false_lock'] = df[fl_col].astype(bool).values if has_fl else False
        melted_parts.append(part)

    melted = pd.concat(melted_parts, ignore_index=True)
    melted['true_dist_cm'] = dist_cm

    return melted[['time_seconds', 'distance_cm', 'is_false_lock',
                    'true_dist_cm']].reset_index(drop=True)


def load_raw_center_readings(grams, dist_cm, trial):
    """Return (t_s_array, {cm_col: arr}) for one trial's raw center cell readings."""
    filename = f"VL5_{grams}g_TRIAL{trial}_{dist_cm}cm.csv"
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    df = df[df['phase_type'] == 'measurement'].copy()
    if len(df) == 0:
        return None, None
    t_s = phase_time_seconds(df)
    vals = {}
    for col in CENTER_CM_COLS:
        if col in df.columns:
            vals[col] = df[col].values
    return t_s, vals


def pool_experiment(all_data, loading_g, dist_cm):
    """Pool all available trials for one (loading, distance) into a single DataFrame."""
    frames = []
    for trial in TRIALS:
        key = (loading_g, trial, dist_cm)
        if key in all_data and all_data[key] is not None:
            frames.append(all_data[key])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_model_stochastic(dist_cm, mass_g, t_grid_s, n_frames=N_FRAMES_ANALYSIS, params=DEFAULT_PARAMS):
    """Stochastic model simulation; returns {t_s, distances_m, distances_cm, is_false_lock}."""
    t_grid_s = np.asarray(t_grid_s, dtype=float)
    D_wall = dist_cm / 100.0
    t_min = t_grid_s / 60.0

    psd = build_psd(mass_g)
    dust_data = build_dust_data(mass_g, params=params)

    data_m = time_series(D_wall, t_min, mass_g,
                         n_frames=n_frames, psd=psd, dust_data=dust_data,
                         params=params,
                         seed=int(round(mass_g * 1000)) + int(dist_cm))
    data_cm = data_m * 100.0
    threshold_cm = dist_cm * FL_THRESHOLD_FRAC

    is_fl = np.where(np.isnan(data_cm), False, data_cm < threshold_cm)

    return {
        't_s': t_grid_s,
        'distances_m': data_m,
        'distances_cm': data_cm,
        'is_false_lock': is_fl,
    }


def model_stochastic(dist_cm, mass_g, t_grid_s, n_frames=N_FRAMES_ANALYSIS, params=DEFAULT_PARAMS):
    """Return (n_times, n_frames) array of model distances in cm."""
    result = run_model_stochastic(dist_cm, mass_g, t_grid_s, n_frames, params=params)
    return result['distances_cm']


def make_model_time_grid():
    """Dense 0-2 min @ 0.5s + sparse 2-15 min @ 2s, in seconds."""
    t_dense = np.arange(0, 120, 0.5)
    t_sparse = np.arange(120, 900 + 2, 2)
    return np.unique(np.concatenate([t_dense, t_sparse]))


def make_time_bins(t_max_s):
    """Return (bin_edges, bin_centers) arrays for 30s bins."""
    edges = np.arange(0, t_max_s + TIME_BIN_WIDTH_S, TIME_BIN_WIDTH_S)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers


def snap_time_from_fl_series(time_centers_s, f_fl_values, threshold=0.05,
                              n_confirm=3):
    """Snap-to-wall time from f_FL(t); returns (snap_time_s, is_always_wall, is_never_snapped)."""
    is_wall = f_fl_values < threshold

    if len(is_wall) == 0:
        return np.nan, False, True

    if np.all(is_wall):
        return 0.0, True, False

    for i in range(len(is_wall) - 1, -1, -1):
        if not is_wall[i]:
            candidate = i + 1
            if candidate + n_confirm <= len(is_wall):
                return time_centers_s[candidate], False, False
            else:
                return np.nan, False, True

    return np.nan, False, True


def _bin_stats_exp(pooled, t_lo, t_hi, dist_cm):
    """Compute per-bin stats from experimental data."""
    mask = ((pooled['time_seconds'] >= t_lo) &
            (pooled['time_seconds'] < t_hi))
    chunk = pooled.loc[mask]

    n_total = len(chunk)
    if n_total == 0:
        return None

    dists = chunk['distance_cm'].dropna()
    fl = chunk['is_false_lock']

    n_fl = int(fl.sum())
    n_wall = n_total - n_fl

    row = {
        'n_readings_exp': n_total,
        'n_wall_exp': n_wall,
        'n_fl_exp': n_fl,
        'f_fl_exp': n_fl / n_total if n_total > 0 else np.nan,
    }

    wall_dists = chunk.loc[~chunk['is_false_lock'], 'distance_cm'].dropna()
    if len(wall_dists) > 0:
        bias = wall_dists - dist_cm
        row['bias_wall_mean_exp'] = bias.mean()
        row['bias_wall_p25_exp'] = np.percentile(bias, 25)
        row['bias_wall_p75_exp'] = np.percentile(bias, 75)
        row['dist_wall_mean_exp'] = wall_dists.mean()
    else:
        row['bias_wall_mean_exp'] = np.nan
        row['bias_wall_p25_exp'] = np.nan
        row['bias_wall_p75_exp'] = np.nan
        row['dist_wall_mean_exp'] = np.nan

    all_dists = dists.values
    row['dist_all_mean_exp'] = all_dists.mean() if len(all_dists) > 0 else np.nan
    row['_vals_exp'] = all_dists

    return row


def _bin_stats_mod(model_result, t_lo, t_hi, dist_cm):
    """Compute per-bin stats from model data."""
    time_mask = ((model_result['t_s'] >= t_lo) &
                 (model_result['t_s'] < t_hi))

    if not np.any(time_mask):
        return None

    bin_distances = model_result['distances_cm'][time_mask, :].ravel()
    bin_fl = model_result['is_false_lock'][time_mask, :].ravel()

    valid = ~np.isnan(bin_distances)
    bin_distances = bin_distances[valid]
    bin_fl = bin_fl[valid]

    n_total = len(bin_distances)
    if n_total == 0:
        return None

    n_fl = int(bin_fl.sum())
    n_wall = n_total - n_fl

    row = {
        'n_readings_mod': n_total,
        'n_wall_mod': n_wall,
        'n_fl_mod': n_fl,
        'f_fl_mod': n_fl / n_total if n_total > 0 else np.nan,
    }

    wall_dists = bin_distances[~bin_fl]
    if len(wall_dists) > 0:
        bias = wall_dists - dist_cm
        row['bias_wall_mean_mod'] = np.mean(bias)
        row['bias_wall_p25_mod'] = np.percentile(bias, 25)
        row['bias_wall_p75_mod'] = np.percentile(bias, 75)
        row['dist_wall_mean_mod'] = np.mean(wall_dists)
    else:
        row['bias_wall_mean_mod'] = np.nan
        row['bias_wall_p25_mod'] = np.nan
        row['bias_wall_p75_mod'] = np.nan
        row['dist_wall_mean_mod'] = np.nan

    row['dist_all_mean_mod'] = np.mean(bin_distances) if n_total > 0 else np.nan
    row['_vals_mod'] = bin_distances

    return row


def compute_timeseries_metrics(all_data, loading_g):
    """Compute parallel time-binned metrics for all distances at one loading."""
    t_grid = make_model_time_grid()
    bin_edges, bin_centers = make_time_bins(900.0)
    rows = []

    for dist_cm in DISTANCES_CM:
        pooled = pool_experiment(all_data, loading_g, dist_cm)
        if len(pooled) == 0:
            print(f"    {dist_cm}cm: no experimental data, skipping")
            continue

        print(f"    {dist_cm}cm: running model ({len(t_grid)} pts x "
              f"{N_FRAMES_ANALYSIS} frames)...", end='', flush=True)
        model_result = run_model_stochastic(dist_cm, loading_g, t_grid,
                                            n_frames=N_FRAMES_ANALYSIS)
        print(" done")

        for i in range(len(bin_edges) - 1):
            t_lo = bin_edges[i]
            t_hi = bin_edges[i + 1]
            tc = bin_centers[i]

            exp = _bin_stats_exp(pooled, t_lo, t_hi, dist_cm)
            if exp is None:
                continue

            mod = _bin_stats_mod(model_result, t_lo, t_hi, dist_cm)

            row = {
                'loading_g': loading_g,
                'dist_cm': dist_cm,
                'time_bin_center_s': tc,
                'time_bin_center_min': tc / 60.0,
            }

            vals_exp = exp.pop('_vals_exp')
            row.update(exp)

            if mod is not None:
                vals_mod = mod.pop('_vals_mod')
                row.update(mod)
            else:
                for k in ['n_readings_mod', 'n_wall_mod', 'n_fl_mod', 'f_fl_mod',
                           'bias_wall_mean_mod', 'bias_wall_p25_mod',
                           'bias_wall_p75_mod', 'dist_wall_mean_mod',
                           'dist_all_mean_mod']:
                    row[k] = np.nan
                vals_mod = np.array([])

            row['f_fl_diff'] = (row.get('f_fl_mod', np.nan) -
                                row.get('f_fl_exp', np.nan))
            row['bias_wall_diff_cm'] = (row.get('bias_wall_mean_mod', np.nan) -
                                        row.get('bias_wall_mean_exp', np.nan))

            if len(vals_exp) >= 5 and len(vals_mod) >= 5:
                row['wasserstein_cm'] = _wasserstein(vals_exp, vals_mod)
            else:
                row['wasserstein_cm'] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def _per_trial_metrics(all_data, loading_g, dist_cm):
    """Per-trial {settled_biases, snap_times_s, stds} for one (loading, distance)."""
    settled_biases = []
    snap_times_s = []
    stds = []

    for trial in TRIALS:
        key = (loading_g, trial, dist_cm)
        df = all_data.get(key)
        if df is None or len(df) == 0:
            continue

        steady = df[(df['time_seconds'] >= STEADY_STATE_START_S) &
                    ~df['is_false_lock']]
        vals = steady['distance_cm'].dropna()
        if len(vals) >= 10:
            settled_biases.append(np.mean(vals) - dist_cm)
            stds.append(np.std(vals))

        t_max = df['time_seconds'].max()
        bin_edges = np.arange(0, t_max + TIME_BIN_WIDTH_S, TIME_BIN_WIDTH_S)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        fl_fracs = []
        for i in range(len(bin_edges) - 1):
            mask = ((df['time_seconds'] >= bin_edges[i]) &
                    (df['time_seconds'] < bin_edges[i + 1]))
            chunk = df.loc[mask]
            if len(chunk) > 0:
                fl_fracs.append(chunk['is_false_lock'].mean())
            else:
                fl_fracs.append(np.nan)

        fl_arr = np.array(fl_fracs)
        snap_t, is_awl, is_ns = snap_time_from_fl_series(
            bin_centers, fl_arr, threshold=0.05, n_confirm=3)
        snap_times_s.append(snap_t)

    return {
        'settled_biases': settled_biases,
        'snap_times_s': snap_times_s,
        'stds': stds,
    }


def compute_scalar_metrics(ts_metrics, all_data, loading_g):
    """Collapse timeseries metrics into one row per (loading, distance)."""
    rows = []

    for dist_cm in DISTANCES_CM:
        sub = ts_metrics[(ts_metrics['loading_g'] == loading_g) &
                         (ts_metrics['dist_cm'] == dist_cm)]
        if len(sub) == 0:
            continue

        row = {'loading_g': loading_g, 'dist_cm': dist_cm}

        n_exp = sub['n_readings_exp'].sum()
        n_fl_exp = sub['n_fl_exp'].sum()
        row['f_fl_total_exp'] = n_fl_exp / n_exp if n_exp > 0 else np.nan

        n_mod = sub['n_readings_mod'].sum()
        n_fl_mod = sub['n_fl_mod'].sum()
        row['f_fl_total_mod'] = n_fl_mod / n_mod if n_mod > 0 else np.nan

        first_bin = sub[sub['time_bin_center_s'] <= 30]
        if len(first_bin) > 0:
            n_r = first_bin['n_readings_exp'].sum()
            n_f = first_bin['n_fl_exp'].sum()
            row['f_fl_initial_exp'] = n_f / n_r if n_r > 0 else np.nan
            n_rm = first_bin['n_readings_mod'].sum()
            n_fm = first_bin['n_fl_mod'].sum()
            row['f_fl_initial_mod'] = n_fm / n_rm if n_rm > 0 else np.nan
        else:
            row['f_fl_initial_exp'] = np.nan
            row['f_fl_initial_mod'] = np.nan

        first_5min = sub[sub['time_bin_center_s'] <= 300]
        if len(first_5min) > 0:
            n_r = first_5min['n_readings_exp'].sum()
            n_f = first_5min['n_fl_exp'].sum()
            row['f_fl_5min_exp'] = n_f / n_r if n_r > 0 else np.nan
            n_rm = first_5min['n_readings_mod'].sum()
            n_fm = first_5min['n_fl_mod'].sum()
            row['f_fl_5min_mod'] = n_fm / n_rm if n_rm > 0 else np.nan
        else:
            row['f_fl_5min_exp'] = np.nan
            row['f_fl_5min_mod'] = np.nan

        tc = sub['time_bin_center_s'].values
        f_fl_exp_arr = sub['f_fl_exp'].values
        f_fl_mod_arr = sub['f_fl_mod'].values

        snap_exp, awl_exp, ns_exp = snap_time_from_fl_series(tc, f_fl_exp_arr)
        snap_mod, awl_mod, ns_mod = snap_time_from_fl_series(tc, f_fl_mod_arr)

        row['snap_time_exp_s'] = snap_exp
        row['snap_time_mod_s'] = snap_mod
        row['snap_time_exp_min'] = snap_exp / 60.0 if not np.isnan(snap_exp) else np.nan
        row['snap_time_mod_min'] = snap_mod / 60.0 if not np.isnan(snap_mod) else np.nan
        row['is_always_wall_locked_exp'] = awl_exp
        row['is_always_wall_locked_mod'] = awl_mod
        row['is_never_snapped_exp'] = ns_exp
        row['is_never_snapped_mod'] = ns_mod

        mask_exp = sub['bias_wall_mean_exp'].notna()
        mask_mod = sub['bias_wall_mean_mod'].notna()

        row['bias_wall_mean_exp_cm'] = (sub.loc[mask_exp, 'bias_wall_mean_exp'].mean()
                                        if mask_exp.any() else np.nan)
        row['bias_wall_mean_mod_cm'] = (sub.loc[mask_mod, 'bias_wall_mean_mod'].mean()
                                        if mask_mod.any() else np.nan)

        both = mask_exp & mask_mod
        if both.any():
            diff = (sub.loc[both, 'bias_wall_mean_mod'].values -
                    sub.loc[both, 'bias_wall_mean_exp'].values)
            row['bias_wall_rmse_cm'] = np.sqrt(np.mean(diff**2))
        else:
            row['bias_wall_rmse_cm'] = np.nan

        early_bias = sub[sub['time_bin_center_s'] <= 180]
        if len(early_bias) > 0 and early_bias['bias_wall_mean_exp'].notna().any():
            row['peak_neg_bias_exp_cm'] = early_bias['bias_wall_mean_exp'].min()
            row['peak_neg_bias_exp_pct'] = row['peak_neg_bias_exp_cm'] / dist_cm * 100
        else:
            row['peak_neg_bias_exp_cm'] = np.nan
            row['peak_neg_bias_exp_pct'] = np.nan

        late_bias = sub[sub['time_bin_center_s'] >= 600]
        if len(late_bias) > 0 and late_bias['bias_wall_mean_exp'].notna().any():
            row['settled_bias_exp_cm'] = late_bias['bias_wall_mean_exp'].mean()
            row['settled_bias_exp_pct'] = row['settled_bias_exp_cm'] / dist_cm * 100
        else:
            row['settled_bias_exp_cm'] = np.nan
            row['settled_bias_exp_pct'] = np.nan

        pooled = pool_experiment(all_data, loading_g, dist_cm)
        if len(pooled) > 0:
            steady_exp = pooled[(pooled['time_seconds'] >= STEADY_STATE_START_S) &
                                ~pooled['is_false_lock']]
            vals = steady_exp['distance_cm'].dropna()
            if len(vals) >= 10:
                row['steady_sd_exp_cm'] = np.std(vals)
                row['steady_bias_pooled_exp_cm'] = np.mean(vals) - dist_cm
            else:
                row['steady_sd_exp_cm'] = np.nan
                row['steady_bias_pooled_exp_cm'] = np.nan
        else:
            row['steady_sd_exp_cm'] = np.nan
            row['steady_bias_pooled_exp_cm'] = np.nan

        steady_mod = sub[sub['time_bin_center_s'] >= STEADY_STATE_START_S]
        if len(steady_mod) > 0 and steady_mod['dist_wall_mean_mod'].notna().any():
            p25s = steady_mod['bias_wall_p25_mod'].dropna()
            p75s = steady_mod['bias_wall_p75_mod'].dropna()
            if len(p25s) > 0 and len(p75s) > 0:
                row['steady_sd_mod_cm'] = (np.median(p75s) - np.median(p25s)) / 1.35
            else:
                row['steady_sd_mod_cm'] = np.nan
            biases = steady_mod['bias_wall_mean_mod'].dropna()
            row['steady_bias_mod_cm'] = np.median(biases) if len(biases) > 0 else np.nan
        else:
            row['steady_sd_mod_cm'] = np.nan
            row['steady_bias_mod_cm'] = np.nan

        trial_data = _per_trial_metrics(all_data, loading_g, dist_cm)

        if len(trial_data['settled_biases']) >= 2:
            row['settled_bias_std_cm'] = np.std(trial_data['settled_biases'], ddof=1)
        else:
            row['settled_bias_std_cm'] = np.nan

        valid_snaps = [s for s in trial_data['snap_times_s'] if not np.isnan(s)]
        if len(valid_snaps) >= 2:
            row['snap_time_std_s'] = np.std(valid_snaps, ddof=1)
        elif (len(valid_snaps) == 1 or
              (len(trial_data['snap_times_s']) > 0 and
               all(s == 0 for s in trial_data['snap_times_s'] if not np.isnan(s)))):
            row['snap_time_std_s'] = 0.0
        else:
            row['snap_time_std_s'] = np.nan

        if len(trial_data['stds']) >= 2:
            row['steady_sd_per_trial_mean_cm'] = np.mean(trial_data['stds'])
            row['steady_sd_per_trial_std_cm'] = np.std(trial_data['stds'], ddof=1)
        else:
            row['steady_sd_per_trial_mean_cm'] = np.nan
            row['steady_sd_per_trial_std_cm'] = np.nan

        w1 = sub['wasserstein_cm'].dropna()
        row['wasserstein_mean_cm'] = w1.mean() if len(w1) > 0 else np.nan

        early_w = sub[sub['time_bin_center_s'] <= 120.0]['wasserstein_cm'].dropna()
        row['wasserstein_early_cm'] = early_w.mean() if len(early_w) > 0 else np.nan

        late_w = sub[sub['time_bin_center_s'] >= 300.0]['wasserstein_cm'].dropna()
        row['wasserstein_late_cm'] = late_w.mean() if len(late_w) > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(scalar_metrics):
    """Print formatted summary table to console."""
    for grams in LOADINGS_G:
        sub = scalar_metrics[scalar_metrics['loading_g'] == grams].sort_values('dist_cm')
        if len(sub) == 0:
            continue

        print(f"\n{'='*90}")
        print(f"  {grams} g -- Model vs Experiment Summary")
        print(f"{'='*90}")
        print(f"  {'Dist':>5s}  {'FL_init':>7s}  {'FL_5min':>7s}  {'t_snap':>9s}  "
              f"{'peak_bias':>10s}  {'settled':>9s}  {'s_trial':>8s}  {'bRMSE':>6s}")

        for _, r in sub.iterrows():
            d = int(r['dist_cm'])

            fl_init = r.get('f_fl_initial_exp', np.nan)
            fl_init_s = f"{fl_init*100:.1f}%" if not np.isnan(fl_init) else '--'

            fl_5m = r.get('f_fl_5min_exp', np.nan)
            fl_5m_s = f"{fl_5m*100:.1f}%" if not np.isnan(fl_5m) else '--'

            if r.get('is_always_wall_locked_exp', False):
                s_exp = 'wall'
            elif r.get('is_never_snapped_exp', False):
                s_exp = 'never'
            elif np.isnan(r.get('snap_time_exp_min', np.nan)):
                s_exp = '--'
            else:
                s_exp = f"{r['snap_time_exp_min']:.1f}min"

            pnb = r.get('peak_neg_bias_exp_cm', np.nan)
            pnb_s = f"{pnb:+.3f}cm" if not np.isnan(pnb) else '--'

            sb = r.get('settled_bias_exp_cm', np.nan)
            sb_s = f"{sb:+.3f}cm" if not np.isnan(sb) else '--'

            st = r.get('settled_bias_std_cm', np.nan)
            st_s = f"+/-{st:.3f}" if not np.isnan(st) else '--'

            brmse = r.get('bias_wall_rmse_cm', np.nan)
            brmse_s = f"{brmse:.2f}" if not np.isnan(brmse) else '--'

            print(f"  {d:3d}cm  {fl_init_s:>7s}  {fl_5m_s:>7s}  {s_exp:>9s}  "
                  f"{pnb_s:>10s}  {sb_s:>9s}  {st_s:>8s}  {brmse_s:>6s}")

    print()


def run_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading experimental data...")
    all_data = {}
    for grams in LOADINGS_G:
        for trial in TRIALS:
            for dist_cm in DISTANCES_CM:
                key = (grams, trial, dist_cm)
                all_data[key] = load_experiment(grams, trial, dist_cm)

    n_loaded = sum(1 for v in all_data.values() if v is not None)
    print(f"  Loaded {n_loaded}/{len(all_data)} files")

    print("\nComputing time-binned metrics...")
    ts_frames = []
    for grams in LOADINGS_G:
        print(f"  [{grams}g]")
        ts = compute_timeseries_metrics(all_data, grams)
        ts_frames.append(ts)
    ts_metrics = pd.concat(ts_frames, ignore_index=True)

    print("\nComputing scalar metrics...")
    scalar_frames = []
    for grams in LOADINGS_G:
        sc = compute_scalar_metrics(ts_metrics, all_data, grams)
        scalar_frames.append(sc)
    scalar_metrics = pd.concat(scalar_frames, ignore_index=True)

    ts_path = os.path.join(OUTPUT_DIR, 'timeseries_metrics.csv')
    ts_metrics.to_csv(ts_path, index=False, float_format='%.4f')
    print(f"\nSaved {ts_path}")

    sc_path = os.path.join(OUTPUT_DIR, 'scalar_metrics.csv')
    scalar_metrics.to_csv(sc_path, index=False, float_format='%.4f')
    print(f"Saved {sc_path}")

    print_summary(scalar_metrics)

    return {
        'timeseries_metrics': ts_metrics,
        'scalar_metrics': scalar_metrics,
        'all_data': all_data,
    }


__all__ = [
    'run_analysis',
    'phase_time_seconds',
    'load_experiment',
    'load_raw_center_readings',
    'pool_experiment',
    'run_model_stochastic',
    'model_stochastic',
    'compute_timeseries_metrics',
    'compute_scalar_metrics',
    'DISTANCES_CM', 'LOADINGS_G', 'TRIALS',
    'FL_THRESHOLD_FRAC', 'TIME_BIN_WIDTH_S', 'STEADY_STATE_START_S',
    'DATA_DIR', 'OUTPUT_DIR',
]


if __name__ == '__main__':
    run_analysis()
