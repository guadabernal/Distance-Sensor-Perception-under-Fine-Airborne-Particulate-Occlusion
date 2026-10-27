"""Cross-sensor statistical analysis; writes all statistical_analysis/metrics CSVs."""

import os

from run_level_metrics import (
    load_all_runs, build_run_level_metrics, summarize_conditions, OUTPUT_DIR,
)
from calibration_analysis import load_luts, compute_fits, build_calibration_tables
from baseline_analysis import build_baseline_trial_metrics, build_baseline_summary
from sensor_comparison import (
    build_factorial_tests, build_sensor_contrasts, build_permutation_tests,
)
from false_lock_mechanism import build_mechanism_tables


def print_summary(fits, condition, contrasts, permutation):
    print(f"\nIR LUT fit: A = {fits['ir_A']:.4f}, B = {fits['ir_B']:.4f}")
    print(f"VL5 LUT fit: a = {fits['vl5_slope']:.4f}, b = {fits['vl5_intercept_cm']:.4f} cm")

    vl5 = condition[(condition['sensor_code'] == 'VL5') & (condition['window'] == 'initial_30s')
                    & (condition['loading_g'] == 0.8)]
    print('\ndToF 0.8 g initial 30 s false lock (mean +/- SD over runs):')
    for _, r in vl5.iterrows():
        print(f"  {r['distance_cm']:3d} cm  {100 * r['false_lock_mean']:5.1f} "
              f"+/- {100 * r['false_lock_sd']:4.1f} %")

    print('\nGM nMAE ratio dToF/IR [95% CI], Holm p:')
    for _, r in contrasts.iterrows():
        print(f"  {r['window']:11s} {r['loading_g']}g {r['distance_cm']:2d}cm  "
              f"{r['ratio_dtof_over_ir']:7.3f} [{r['ratio_ci95_low']:.3f}, "
              f"{r['ratio_ci95_high']:.3f}]  p={r['p_value_holm']:.2e}")

    for _, r in permutation.iterrows():
        print(f"\nPermutation test [{r['window']}]: F = {r['observed_partial_f']:.2f}, "
              f"p = {r['p_value']:.2e} (B = {r['n_permutations']})")


def run_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Calibration fits...')
    ir_lut, vl5_lut = load_luts()
    fits = compute_fits(ir_lut, vl5_lut)
    fit_summary, residuals = build_calibration_tables(ir_lut, vl5_lut, fits)

    print('Loading processed runs...')
    runs = load_all_runs()

    print('Baseline transfer and precision...')
    baseline_trial = build_baseline_trial_metrics(runs, fits)

    print('Run-level metrics...')
    run_df = build_run_level_metrics(runs)
    condition = summarize_conditions(run_df)

    print('Sensor comparison (factorial, contrasts, permutation)...')
    contrasts = build_sensor_contrasts(run_df)
    permutation = build_permutation_tests(run_df)

    print('False-lock mechanism tables...')
    tables = {
        'calibration_fit_summary': fit_summary,
        'calibration_residuals': residuals,
        'baseline_trial_metrics': baseline_trial,
        'baseline_summary': build_baseline_summary(baseline_trial),
        'run_level_metrics': run_df,
        'condition_summary': condition,
        'factorial_tests': build_factorial_tests(run_df),
        'sensor_contrasts': contrasts,
        'permutation_tests': permutation,
        **build_mechanism_tables(runs),
    }
    for name, df in tables.items():
        path = os.path.join(OUTPUT_DIR, f'{name}.csv')
        df.to_csv(path, index=False)
        print(f'Saved {path}')

    print_summary(fits, condition, contrasts, permutation)
    return tables


if __name__ == '__main__':
    run_analysis()
