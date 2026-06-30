"""Generate raw-phase and calibrated debug figures for every experiment."""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_pre_processing import (
    load_ir_data, load_vl5_data, parse_filename,
    VL5_CENTER_2X2, CONTROL_WINDOW_S, CONTROL_MARGIN_S,
    RAW_DATA_DIR, PROCESSED_DIR,
)


VL5_CENTER_LABELS = ['Cell (3,3)', 'Cell (3,4)', 'Cell (4,3)', 'Cell (4,4)']
VL5_CENTER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
SENSOR_Y_LABELS = {'IR': 'Voltage (V)', 'VL5': 'Distance (mm)'}

_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'debugFigures')


def compute_phase_timestamps(df_raw):
    """Return key event timestamps (ms) from a raw DataFrame with all phases."""
    max_phase = int(df_raw['phase'].max())
    control_phase = 2 if max_phase == 7 else 1
    dust_phase = max_phase
    return {
        't_control_ms': df_raw.loc[df_raw['phase'] == control_phase, 'timestamp'].iloc[0],
        't_injection_ms': df_raw.loc[df_raw['phase'] == control_phase + 1, 'timestamp'].iloc[0],
        't_measurement_ms': df_raw.loc[df_raw['phase'] == dust_phase, 'timestamp'].iloc[0],
    }


def generate_debug_figures():
    raw_output_dir = os.path.join(_OUTPUT_DIR, 'raw_phase_readings')
    cal_output_dir = os.path.join(_OUTPUT_DIR, 'calibrated_distances')

    folder_configs = {
        'IR2gData': ('IR', load_ir_data),
        'IR8gData': ('IR', load_ir_data),
        'VL52gData': ('VL5', load_vl5_data),
        'VL58gData': ('VL5', load_vl5_data),
    }

    for folder, (sensor_type, loader) in folder_configs.items():
        folder_path = os.path.join(RAW_DATA_DIR, folder)
        csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
        print(f"Generating plots for {folder} ({len(csv_files)} files)...")

        for filepath in csv_files:
            filename = os.path.basename(filepath)
            metadata = parse_filename(filename)
            trial = metadata['trial']
            grams = metadata['grams']
            distance_cm = metadata['distance_cm']
            base_name = f"{sensor_type}_{grams}g_TRIAL{trial}_{distance_cm}cm"
            label = f"{sensor_type} {grams}g Trial {trial} {distance_cm}cm"

            df_raw = loader(filepath)
            timestamps = compute_phase_timestamps(df_raw)
            t_ref = df_raw['timestamp'].iloc[0]
            df_raw['t_s'] = (df_raw['timestamp'] - t_ref) / 1000.0

            events = [
                ((timestamps['t_control_ms'] - t_ref) / 1000.0, 'Control start'),
                ((timestamps['t_injection_ms'] - t_ref) / 1000.0, 'Injection start'),
                ((timestamps['t_measurement_ms'] - t_ref) / 1000.0, 'Measurement start'),
            ]

            raw_out_dir = os.path.join(raw_output_dir, sensor_type,
                                       f'{grams}g', f'TRIAL{trial}')
            os.makedirs(raw_out_dir, exist_ok=True)

            fig, ax = plt.subplots(figsize=(14, 6))
            phases = sorted(df_raw['phase'].astype(int).unique())
            cmap = plt.cm.tab10

            if sensor_type == 'VL5':
                all_cell_cols = [f'cell_{i}' for i in range(64)]
                for phase in phases:
                    mask = df_raw['phase'] == phase
                    t_vals = df_raw.loc[mask, 't_s'].values
                    cell_vals = df_raw.loc[mask, all_cell_cols].values
                    t_rep = np.repeat(t_vals, 64)
                    v_rep = cell_vals.ravel()
                    ax.scatter(t_rep, v_rep, c=[cmap(phase % 10)], s=1,
                               alpha=0.3, label=f'Phase {phase}')
            else:
                for phase in phases:
                    mask = df_raw['phase'] == phase
                    ax.scatter(df_raw.loc[mask, 't_s'], df_raw.loc[mask, 'value'],
                               c=[cmap(phase % 10)], s=2, alpha=0.5, label=f'Phase {phase}')
            for t, ev_label in events:
                ax.axvline(t, color='k', linestyle='--', linewidth=1)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(SENSOR_Y_LABELS[sensor_type])
            ax.set_title(f'{label} - Raw (all phases)')
            ax.legend(loc='upper right', markerscale=4, fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            for t, ev_label in events:
                ax.text(t, ax.get_ylim()[1], f'  {ev_label}',
                        rotation=90, va='top', fontsize=8)
            fig.savefig(os.path.join(raw_out_dir, f'{base_name}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

            if sensor_type == 'VL5':
                c4_raw_dir = os.path.join(raw_output_dir, 'VL5_Center4',
                                          f'{grams}g', f'TRIAL{trial}')
                os.makedirs(c4_raw_dir, exist_ok=True)

                fig_c4, ax_c4 = plt.subplots(figsize=(14, 6))
                center_cell_cols = [f'cell_{i}' for i in VL5_CENTER_2X2]
                for phase in phases:
                    mask = df_raw['phase'] == phase
                    t_vals = df_raw.loc[mask, 't_s'].values
                    cell_vals = df_raw.loc[mask, center_cell_cols].values
                    t_rep = np.repeat(t_vals, len(VL5_CENTER_2X2))
                    v_rep = cell_vals.ravel()
                    ax_c4.scatter(t_rep, v_rep, c=[cmap(phase % 10)], s=2,
                                  alpha=0.4, label=f'Phase {phase}')
                for t, ev_label in events:
                    ax_c4.axvline(t, color='k', linestyle='--', linewidth=1)
                ax_c4.set_xlabel('Time (s)')
                ax_c4.set_ylabel(SENSOR_Y_LABELS['VL5'])
                ax_c4.set_title(f'{label} - Raw Center 2x2 (all phases)')
                ax_c4.legend(loc='upper right', markerscale=4, fontsize=8)
                ax_c4.grid(True, alpha=0.3)
                plt.tight_layout()
                for t, ev_label in events:
                    ax_c4.text(t, ax_c4.get_ylim()[1], f'  {ev_label}',
                               rotation=90, va='top', fontsize=8)
                fig_c4.savefig(os.path.join(c4_raw_dir, f'{base_name}.png'),
                               dpi=300, bbox_inches='tight')
                plt.close(fig_c4)

            proc_path = os.path.join(PROCESSED_DIR, sensor_type, f'{base_name}.csv')

            cal_out_dir = os.path.join(cal_output_dir, sensor_type,
                                       f'{grams}g', f'TRIAL{trial}')
            os.makedirs(cal_out_dir, exist_ok=True)

            if sensor_type == 'VL5':
                cal_cols = [f'cell_{i}_cm' for i in VL5_CENTER_2X2]
                fl_cols = [f'cell_{i}_fl' for i in VL5_CENTER_2X2]
                vl5_load_cols = ['time_seconds', 'phase_type', 'bias_offset_cm'] + cal_cols
                _tmp_cols = pd.read_csv(proc_path, nrows=0).columns.tolist()
                for fc in fl_cols:
                    if fc in _tmp_cols:
                        vl5_load_cols.append(fc)
                df_proc = pd.read_csv(proc_path, usecols=vl5_load_cols)
            else:
                df_proc = pd.read_csv(proc_path)

            offset = df_proc['bias_offset_cm'].iloc[0]
            meas_title = f'Measurement ({offset:+.2f} cm)'

            fig, (ax_ctrl, ax_meas) = plt.subplots(1, 2, figsize=(18, 6))
            for ax, phase_type, title_suffix in [
                (ax_ctrl, 'control', 'Control'),
                (ax_meas, 'measurement', meas_title),
            ]:
                mask = df_proc['phase_type'] == phase_type
                if not mask.any():
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Calibrated Distance (cm)')
                    ax.set_title(f'{title_suffix} phase')
                    ax.grid(True, alpha=0.3)
                    continue
                sub = df_proc[mask]

                if phase_type == 'control':
                    t_ctrl_end = sub['time_seconds'].max()
                    win_end = t_ctrl_end - CONTROL_MARGIN_S
                    win_start = win_end - CONTROL_WINDOW_S
                    in_window = ((sub['time_seconds'] >= win_start) &
                                 (sub['time_seconds'] <= win_end))
                    outside = sub[~in_window]
                    inside = sub[in_window]
                    if sensor_type == 'VL5':
                        for col, color, clabel in zip(
                            cal_cols, VL5_CENTER_COLORS, VL5_CENTER_LABELS
                        ):
                            ax.scatter(outside['time_seconds'], outside[col],
                                       c=color, s=2, alpha=0.3)
                            ax.scatter(inside['time_seconds'], inside[col],
                                       c='red', s=4, alpha=0.8)
                    else:
                        if 'is_ghost' in outside.columns:
                            clean_out = outside[~outside['is_ghost']]
                            ghost_out = outside[outside['is_ghost']]
                            ax.scatter(clean_out['time_seconds'], clean_out['distance_cm_calibrated'],
                                       c='#1f77b4', s=2, alpha=0.3)
                            ax.scatter(ghost_out['time_seconds'], ghost_out['distance_cm_calibrated'],
                                       c='#d62728', s=3, alpha=0.4)
                        else:
                            ax.scatter(outside['time_seconds'], outside['distance_cm_calibrated'],
                                       c='#1f77b4', s=2, alpha=0.3)
                        if 'is_ghost' in inside.columns:
                            ghost_in = inside[inside['is_ghost']]
                            clean_in = inside[~inside['is_ghost']]
                            ax.scatter(ghost_in['time_seconds'], ghost_in['distance_cm_calibrated'],
                                       c='red', s=4, alpha=0.5, label='Ghost (filtered)')
                            ax.scatter(clean_in['time_seconds'], clean_in['distance_cm_calibrated'],
                                       c='#2ca02c', s=4, alpha=0.8, label='Used for median')
                            if 'distance_cm_filtered' in inside.columns:
                                median_val = inside['distance_cm_filtered'].median()
                            elif len(clean_in) >= 10:
                                median_val = clean_in['distance_cm_calibrated'].median()
                            else:
                                median_val = inside['distance_cm_calibrated'].median()
                            ax.hlines(median_val, win_start, win_end,
                                      colors='#2ca02c', linewidths=2, label=f'Median = {median_val:.2f} cm')
                            if 'distance_cm_smooth' in inside.columns:
                                ax.plot(inside['time_seconds'], inside['distance_cm_smooth'],
                                        c='#ff8c00', linewidth=1.2, alpha=0.8,
                                        label='Smoothed', zorder=4)
                        else:
                            ax.scatter(inside['time_seconds'], inside['distance_cm_calibrated'],
                                       c='red', s=4, alpha=0.8)
                    ax.axvspan(win_start, win_end, color='#2ca02c',
                               alpha=0.08, label='Bias avg window')
                else:
                    if sensor_type == 'VL5':
                        has_any_fl = False
                        for ci, col, color, clabel in zip(
                            VL5_CENTER_2X2, cal_cols,
                            VL5_CENTER_COLORS, VL5_CENTER_LABELS
                        ):
                            fl_col = f'cell_{ci}_fl'
                            if fl_col in sub.columns:
                                pt_colors = np.where(sub[fl_col],
                                                     '#ff69b4', color)
                                if sub[fl_col].any():
                                    has_any_fl = True
                            else:
                                pt_colors = color
                            ax.scatter(sub['time_seconds'], sub[col],
                                       c=pt_colors, s=2, alpha=0.5)
                        for color, clabel in zip(VL5_CENTER_COLORS,
                                                 VL5_CENTER_LABELS):
                            ax.scatter([], [], c=color, s=2, label=clabel)
                        if has_any_fl:
                            ax.scatter([], [], c='#ff69b4', s=2,
                                       label='False lock')
                        ax.axhline(0.70 * distance_cm, color='#ff69b4',
                                   linewidth=1, alpha=0.4, linestyle='--')
                    else:
                        if 'is_ghost' in sub.columns and 'distance_cm_filtered' in sub.columns:
                            clean = sub[~sub['is_ghost']]
                            ghost = sub[sub['is_ghost']]
                            ax.scatter(clean['time_seconds'], clean['distance_cm_calibrated'],
                                       c='#1f77b4', s=2, alpha=0.4, label='Clean raw', zorder=2)
                            ax.scatter(ghost['time_seconds'], ghost['distance_cm_calibrated'],
                                       c='#d62728', s=3, alpha=0.5, label=f'Ghost ({len(ghost)})', zorder=3)
                            ax.plot(sub['time_seconds'], sub['distance_cm_filtered'],
                                    c='#2ca02c', linewidth=0.8, alpha=0.5,
                                    label='Filtered (median)', zorder=4)
                            if 'distance_cm_smooth' in sub.columns:
                                ax.plot(sub['time_seconds'], sub['distance_cm_smooth'],
                                        c='#ff8c00', linewidth=1.5, alpha=0.95,
                                        label='Smoothed (Savitzky-Golay)', zorder=5)
                            ax.legend(loc='lower right', markerscale=4, fontsize=7)
                        else:
                            ax.scatter(sub['time_seconds'], sub['distance_cm_calibrated'],
                                       c='#d62728', s=2, alpha=0.5)

                if phase_type == 'control' or sensor_type == 'VL5':
                    ax.legend(loc='upper right', markerscale=2, fontsize=7)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Calibrated Distance (cm)')
                ax.set_title(f'{title_suffix} phase')
                ax.grid(True, alpha=0.3)
            fig.suptitle(f'{label} - Calibrated', fontsize=13)
            plt.tight_layout()
            fig.savefig(os.path.join(cal_out_dir, f'{base_name}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

            if sensor_type == 'VL5':
                c4_cal_dir = os.path.join(cal_output_dir, 'VL5_Center4',
                                          f'{grams}g', f'TRIAL{trial}')
                os.makedirs(c4_cal_dir, exist_ok=True)

                c4_cal_cols = [f'cell_{i}_cm' for i in VL5_CENTER_2X2]
                fig_c4, (ax_ctrl_c4, ax_meas_c4) = plt.subplots(1, 2, figsize=(18, 6))
                for ax_c4, phase_type, title_suffix in [
                    (ax_ctrl_c4, 'control', 'Control'),
                    (ax_meas_c4, 'measurement', meas_title),
                ]:
                    mask = df_proc['phase_type'] == phase_type
                    if not mask.any():
                        ax_c4.set_xlabel('Time (s)')
                        ax_c4.set_ylabel('Calibrated Distance (cm)')
                        ax_c4.set_title(f'{title_suffix} phase')
                        ax_c4.grid(True, alpha=0.3)
                        continue
                    sub = df_proc[mask]

                    if phase_type == 'control':
                        t_ctrl_end = sub['time_seconds'].max()
                        win_end = t_ctrl_end - CONTROL_MARGIN_S
                        win_start = win_end - CONTROL_WINDOW_S
                        in_window = ((sub['time_seconds'] >= win_start) &
                                     (sub['time_seconds'] <= win_end))
                        outside = sub[~in_window]
                        inside = sub[in_window]
                        for col, color, clabel in zip(
                            c4_cal_cols, VL5_CENTER_COLORS, VL5_CENTER_LABELS
                        ):
                            ax_c4.scatter(outside['time_seconds'], outside[col],
                                          c=color, s=2, alpha=0.3)
                            ax_c4.scatter(inside['time_seconds'], inside[col],
                                          c='red', s=4, alpha=0.8)
                        ax_c4.axvspan(win_start, win_end, color='red',
                                      alpha=0.08, label='Bias avg window')
                    else:
                        has_any_fl = False
                        for ci, col, color, clabel in zip(
                            VL5_CENTER_2X2, c4_cal_cols,
                            VL5_CENTER_COLORS, VL5_CENTER_LABELS
                        ):
                            fl_col = f'cell_{ci}_fl'
                            if fl_col in sub.columns:
                                pt_colors = np.where(sub[fl_col],
                                                     '#ff69b4', color)
                                if sub[fl_col].any():
                                    has_any_fl = True
                            else:
                                pt_colors = color
                            ax_c4.scatter(sub['time_seconds'], sub[col],
                                          c=pt_colors, s=2, alpha=0.5)
                        for color, clabel in zip(VL5_CENTER_COLORS,
                                                 VL5_CENTER_LABELS):
                            ax_c4.scatter([], [], c=color, s=2, label=clabel)
                        if has_any_fl:
                            ax_c4.scatter([], [], c='#ff69b4', s=2,
                                          label='False lock')
                        ax_c4.axhline(0.70 * distance_cm, color='#ff69b4',
                                      linewidth=1, alpha=0.4, linestyle='--')

                    ax_c4.legend(loc='upper right', markerscale=2, fontsize=7)
                    ax_c4.set_xlabel('Time (s)')
                    ax_c4.set_ylabel('Calibrated Distance (cm)')
                    ax_c4.set_title(f'{title_suffix} phase')
                    ax_c4.grid(True, alpha=0.3)
                fig_c4.suptitle(f'{label} - Calibrated Center 2x2', fontsize=13)
                plt.tight_layout()
                fig_c4.savefig(os.path.join(c4_cal_dir, f'{base_name}.png'),
                               dpi=300, bbox_inches='tight')
                plt.close(fig_c4)

            print(f"  {base_name}: raw + calibrated")

    print("Done!")


if __name__ == '__main__':
    generate_debug_figures()
