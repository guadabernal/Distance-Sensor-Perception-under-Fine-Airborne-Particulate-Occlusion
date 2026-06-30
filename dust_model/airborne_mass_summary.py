"""Write total airborne dust mass at t=0 (injection), t=150s (gate opens), t=1050s (meas end)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dust_model import run_dust_model, V_BOX, ETA_REL, SETTLING_TIME_S, DT_S

MEASUREMENT_DURATION_S = 900.0
MASS_LOADINGS_G = [0.2, 0.8]

TIMEPOINTS = {
    'injection':       0.0,
    'gate_opens':      SETTLING_TIME_S,
    'measurement_end': SETTLING_TIME_S + MEASUREMENT_DURATION_S,
}


def summarize(mass_g: float) -> list[dict]:
    """One row per timepoint for a given loading."""
    _, df_ts, C_0 = run_dust_model(mass_g)
    rows = []
    for label, t_s in TIMEPOINTS.items():
        idx = int(round(t_s / DT_S))
        C_mg = float(df_ts['C_total_mg_m3'].iloc[idx])
        M_mg = C_mg * V_BOX
        rows.append({
            'mass_g': mass_g,
            'timepoint': label,
            't_s': float(df_ts['time_s'].iloc[idx]),
            'C_mg_m3': round(C_mg, 2),
            'C_g_m3': round(C_mg * 1e-3, 4),
            'M_mg': round(M_mg, 2),
            'M_g': round(M_mg * 1e-3, 4),
            'frac_remaining': round(C_mg / C_0 if C_0 > 0 else np.nan, 4),
        })
    return rows


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'dust_model_plots')
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for mass_g in MASS_LOADINGS_G:
        rows.extend(summarize(mass_g))
    df = pd.DataFrame(rows)

    out_path = os.path.join(output_dir, 'airborne_mass_summary.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
