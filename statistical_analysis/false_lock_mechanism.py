"""dToF false-lock state mechanism: per-run state occupancies, threshold sweep, false-return ranges, histograms, and occupancy by concentration."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dust_model'))
from dust_model import get_measurement_total_concentration_g_m3
from run_level_metrics import CENTER_CM_COLS, WINDOWS, time_weighted_mean


STATE_NAMES = ('false_lock', 'intermediate', 'wall_like', 'invalid')
FALSE_LOCK_FRAC = 0.70
WALL_LIKE_MIN_FRAC = 0.90
THRESHOLD_SWEEP_FRACS = (0.60, 0.70, 0.80)
FALSE_RETURN_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
HIST_EDGES = np.round(np.arange(0.0, 1.201, 0.01), 10)
CONC_BIN_EDGES_G_M3 = np.round(np.arange(0.5, 9.51, 0.5), 10)
CURVE_SMOOTH_S = 30.0


def classify_readings(cm, true_cm, false_lock_frac=FALSE_LOCK_FRAC):
    """Per-reading state masks from q = D_rep / D; non-finite readings are invalid."""
    cm = np.asarray(cm, dtype=float)
    finite = np.isfinite(cm)
    with np.errstate(invalid='ignore'):
        q = cm / float(true_cm)
        return {'false_lock': finite & (q < false_lock_frac),
                'intermediate': finite & (q >= false_lock_frac) & (q < WALL_LIKE_MIN_FRAC),
                'wall_like': finite & (q >= WALL_LIKE_MIN_FRAC),
                'invalid': ~finite}


def frame_state_fractions(cm, true_cm):
    return {state: mask.mean(axis=1) for state, mask in classify_readings(cm, true_cm).items()}


def frame_abs_error_gt_10pct(cm, true_cm):
    cm = np.asarray(cm, dtype=float)
    D = float(true_cm)
    with np.errstate(invalid='ignore'):
        return (np.isfinite(cm) & (np.abs(cm - D) > 0.10 * D)).mean(axis=1)


def frame_weights(t):
    """Trapezoidal time weight of each frame."""
    w = np.zeros(len(t))
    if len(t) > 1:
        dt = np.diff(t)
        w[:-1] += dt / 2.0
        w[1:] += dt / 2.0
    return w


def run_state_metrics(t, cm, true_cm, start, end):
    """Time-weighted state occupancies and threshold sweep for one run over [start, end]."""
    fractions = frame_state_fractions(cm, true_cm)
    out = {f'{state}_fraction': time_weighted_mean(t, fractions[state], start, end)
           for state in STATE_NAMES}
    out['false_lock_operational_fraction'] = time_weighted_mean(
        t, fractions['false_lock'] + fractions['invalid'], start, end)
    out['frac_abs_error_gt_10pct_time_weighted'] = time_weighted_mean(
        t, frame_abs_error_gt_10pct(cm, true_cm), start, end)
    out['substantial_error_excess_fraction'] = (
        out['frac_abs_error_gt_10pct_time_weighted'] - out['false_lock_fraction'])
    sweep = {frac: time_weighted_mean(
        t, classify_readings(cm, true_cm, frac)['false_lock'].mean(axis=1), start, end)
        for frac in THRESHOLD_SWEEP_FRACS}
    return out, sweep


def weighted_quantiles(values, weights, quantiles):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & (weights > 0)
    if not keep.any():
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values[keep], kind='stable')
    v, w = values[keep][order], weights[keep][order]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(quantiles, cw, v)


def run_false_return_summary(t, cm, true_cm, start, end):
    """Time-weighted quantiles of false-lock reported range (absolute and normalized)."""
    D = float(true_cm)
    inside = (t >= start) & (t <= end)
    cm_in = cm[inside]
    finite = np.isfinite(cm_in)
    n_finite = finite.sum(axis=1)
    per_reading = np.where(n_finite > 0, frame_weights(t[inside]) / np.maximum(n_finite, 1), 0.0)
    weights = np.where(finite, per_reading[:, None], 0.0)
    fs = classify_readings(cm_in, D)['false_lock']
    vals, w = cm_in[fs], weights[fs]
    out = {'n_false_lock_readings': int(fs.sum()),
           'false_lock_weight': float(w.sum()),
           'total_finite_weight': float(weights.sum())}
    abs_q = weighted_quantiles(vals, w, FALSE_RETURN_QUANTILES)
    norm_q = weighted_quantiles(vals / D, w, FALSE_RETURN_QUANTILES)
    for pq, a, n in zip(FALSE_RETURN_QUANTILES, abs_q, norm_q):
        tag = f'p{int(round(pq * 100)):02d}'
        out[f'abs_range_cm_{tag}'] = float(a)
        out[f'q_{tag}'] = float(n)
    return out


def run_normalized_histogram(t, cm, true_cm, start, end):
    """Time-weighted distribution of q = D_rep / D over HIST_EDGES (finite readings only)."""
    finite = np.isfinite(cm)
    n_finite = finite.sum(axis=1)
    n_bins = len(HIST_EDGES) - 1
    with np.errstate(invalid='ignore'):
        idx = np.digitize(np.where(finite, cm / float(true_cm), np.nan), HIST_EDGES, right=False)

    frame_probs = np.zeros((len(t), n_bins + 2))
    for j in range(cm.shape[1]):
        rows = np.flatnonzero(finite[:, j])
        frame_probs[rows, np.clip(idx[rows, j], 0, n_bins + 1)] += 1.0
    frame_probs = np.where(n_finite[:, None] > 0,
                           frame_probs / np.maximum(n_finite, 1)[:, None], 0.0)

    tw = np.array([time_weighted_mean(t, frame_probs[:, k], start, end)
                   for k in range(n_bins + 2)])
    probs = tw / tw.sum() if tw.sum() > 0 else tw
    return {'probs': probs[1:-1], 'p_underflow': float(probs[0]),
            'p_overflow': float(probs[-1]),
            'p_invalid': time_weighted_mean(t, (~finite).mean(axis=1), start, end)}


def centered_time_weighted_rolling_mean(t, y, window_s):
    """Centered rolling mean of a piecewise-linear y(t), integrated exactly over each window."""
    if len(t) == 1:
        return y.copy()
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])

    def integral_to(x):
        i = np.clip(np.searchsorted(t, x, side='right') - 1, 0, len(t) - 2)
        dt = x - t[i]
        y_x = y[i] + (y[i + 1] - y[i]) * np.where(t[i + 1] > t[i], dt / (t[i + 1] - t[i]), 0.0)
        return cum[i] + 0.5 * (y[i] + y_x) * dt

    lo = np.maximum(t[0], t - window_s / 2.0)
    hi = np.minimum(t[-1], t + window_s / 2.0)
    duration = hi - lo
    out = y.copy()
    pos = duration > 0
    out[pos] = (integral_to(hi[pos]) - integral_to(lo[pos])) / duration[pos]
    return out


def run_aware_state_curve(t, cm, true_cm, x_of_t, state='false_lock',
                          smooth_window_s=CURVE_SMOOTH_S):
    """Per-run smoothed state fraction; returns (t, fraction, x_of_t(t))."""
    t = np.asarray(t, dtype=float)
    f = frame_state_fractions(cm, true_cm)[state]
    return t, centered_time_weighted_rolling_mean(t, f, smooth_window_s), np.asarray(x_of_t(t), dtype=float)


def interpolate_runs_to_common_grid(curves, grid):
    """Interpolate (x, y) run curves onto grid (NaN outside each run); returns (matrix, run mean)."""
    rows = []
    for x, y in curves:
        order = np.argsort(x, kind='stable')
        x, y = np.asarray(x, dtype=float)[order], np.asarray(y, dtype=float)[order]
        yi = np.interp(grid, x, y)
        yi[(grid < x[0]) | (grid > x[-1])] = np.nan
        rows.append(yi)
    mat = np.vstack(rows)
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(mat, axis=0)
    mean[np.all(np.isnan(mat), axis=0)] = np.nan
    return mat, mean


def run_occupancy_by_concentration_bin(t, cm, true_cm, conc):
    """Time-weighted state occupancy within each modeled-concentration bin."""
    fractions = frame_state_fractions(cm, true_cm)
    gt10 = frame_abs_error_gt_10pct(cm, true_cm)
    w = frame_weights(t)
    idx = np.digitize(conc, CONC_BIN_EDGES_G_M3, right=False) - 1
    rows = []
    for b in range(len(CONC_BIN_EDGES_G_M3) - 1):
        sel = idx == b
        wsum = w[sel].sum()
        if not sel.any() or wsum <= 0:
            continue
        row = {'conc_bin_low_g_m3': float(CONC_BIN_EDGES_G_M3[b]),
               'conc_bin_high_g_m3': float(CONC_BIN_EDGES_G_M3[b + 1]),
               'n_frames': int(sel.sum()), 'time_in_bin_s': float(wsum)}
        for state in STATE_NAMES:
            row[f'{state}_fraction'] = float(np.sum(fractions[state][sel] * w[sel]) / wsum)
        row['frac_abs_error_gt_10pct'] = float(np.sum(gt10[sel] * w[sel]) / wsum)
        rows.append(row)
    return rows


def _equal_run_stats(values):
    v = np.asarray(values, dtype=float)
    return {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)),
            'min': float(np.min(v)), 'max': float(np.max(v))}


def build_mechanism_tables(runs):
    """All false-lock mechanism tables from the 36 dToF runs, keyed by output CSV stem."""
    run_rows, thr_rows, ret_rows, hist_rows, occ_rows = [], [], [], [], []
    hists, occ_bins = {}, {}

    for (sensor, grams, dist_cm, trial), (_, meas) in runs.items():
        if sensor != 'VL5':
            continue
        t = meas['time_seconds'].values
        cm = meas[CENTER_CM_COLS].values.astype(float)
        conc = get_measurement_total_concentration_g_m3(grams, t)

        for window, (a, b) in WINDOWS.items():
            base = {'window': window, 'loading_g': grams, 'distance_cm': dist_cm,
                    'trial': trial, 'n_frames': int(((t >= a) & (t <= b)).sum())}
            states, sweep = run_state_metrics(t, cm, dist_cm, a, b)
            run_rows.append({**base, **states})
            for frac, val in sweep.items():
                thr_rows.append({**base, 'level': 'run', 'threshold_frac': frac,
                                 'false_lock_fraction': val})
            ret_rows.append({**base, **run_false_return_summary(t, cm, dist_cm, a, b)})
            hist = run_normalized_histogram(t, cm, dist_cm, a, b)
            hists.setdefault((window, grams, dist_cm), []).append(hist)
            for k in range(len(HIST_EDGES) - 1):
                hist_rows.append({**base, 'q_bin_low': float(HIST_EDGES[k]),
                                  'q_bin_high': float(HIST_EDGES[k + 1]),
                                  'probability': float(hist['probs'][k]),
                                  'p_underflow': hist['p_underflow'],
                                  'p_overflow': hist['p_overflow'],
                                  'p_invalid': hist['p_invalid']})

        for row in run_occupancy_by_concentration_bin(t, cm, dist_cm, conc):
            occ_rows.append({'loading_g': grams, 'distance_cm': dist_cm, 'trial': trial,
                             'level': 'run', 'n_runs': 1, **row})
            occ_bins.setdefault((grams, dist_cm, row['conc_bin_low_g_m3'],
                                 row['conc_bin_high_g_m3']), []).append(row)

    run_df = pd.DataFrame(run_rows)
    metric_cols = [c for c in run_df.columns if c.endswith('_fraction')
                   or c == 'frac_abs_error_gt_10pct_time_weighted']
    cond_rows = []
    for (window, grams, dist_cm), grp in run_df.groupby(
            ['window', 'loading_g', 'distance_cm'], sort=True):
        row = {'window': window, 'loading_g': grams, 'distance_cm': dist_cm, 'n_runs': len(grp)}
        for col in metric_cols:
            for stat, val in _equal_run_stats(grp[col].values).items():
                row[f'{col}_{stat}'] = val
        cond_rows.append(row)

    thr_df = pd.DataFrame(thr_rows)
    for (window, grams, dist_cm, frac), grp in thr_df.groupby(
            ['window', 'loading_g', 'distance_cm', 'threshold_frac'], sort=True):
        s = _equal_run_stats(grp['false_lock_fraction'].values)
        thr_rows.append({'window': window, 'loading_g': grams, 'distance_cm': dist_cm,
                         'trial': np.nan, 'n_frames': np.nan, 'level': 'condition_mean',
                         'threshold_frac': frac, 'false_lock_fraction': s['mean'],
                         'false_lock_fraction_sd': s['sd']})

    hist_cond_rows = []
    for (window, grams, dist_cm), hs in sorted(hists.items()):
        probs = np.mean([h['probs'] for h in hs], axis=0)
        for k in range(len(HIST_EDGES) - 1):
            hist_cond_rows.append({
                'window': window, 'loading_g': grams, 'distance_cm': dist_cm,
                'n_runs': len(hs), 'q_bin_low': float(HIST_EDGES[k]),
                'q_bin_high': float(HIST_EDGES[k + 1]), 'probability': float(probs[k]),
                **{p: float(np.mean([h[p] for h in hs]))
                   for p in ('p_underflow', 'p_overflow', 'p_invalid')}})

    for (grams, dist_cm, lo, hi), entries in sorted(occ_bins.items()):
        row = {'loading_g': grams, 'distance_cm': dist_cm, 'trial': np.nan,
               'level': 'condition_mean', 'n_runs': len(entries),
               'conc_bin_low_g_m3': lo, 'conc_bin_high_g_m3': hi,
               'n_frames': int(sum(e['n_frames'] for e in entries)),
               'time_in_bin_s': float(np.mean([e['time_in_bin_s'] for e in entries]))}
        for col in [f'{s}_fraction' for s in STATE_NAMES] + ['frac_abs_error_gt_10pct']:
            row[col] = float(np.mean([e[col] for e in entries]))
        occ_rows.append(row)

    return {
        'false_lock_state_run_metrics': run_df,
        'false_lock_state_condition_summary': pd.DataFrame(cond_rows),
        'false_lock_threshold_sensitivity': pd.DataFrame(thr_rows),
        'false_return_range_run_summary': pd.DataFrame(ret_rows),
        'false_lock_histograms_by_run': pd.DataFrame(hist_rows),
        'false_lock_histograms_condition_summary': pd.DataFrame(hist_cond_rows),
        'false_lock_state_occupancy_vs_concentration': pd.DataFrame(occ_rows),
    }
