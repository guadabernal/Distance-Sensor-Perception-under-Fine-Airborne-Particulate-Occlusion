"""Side-by-side comparison of active calibration vs archived parameter files."""
from __future__ import annotations

import os
import sys
import glob

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from tof_calibrate import read_params_file, ALL_PARAM_NAMES  # noqa: E402

TOF_DIR = os.path.dirname(_THIS_DIR)
METRICS_DIR = os.path.join(TOF_DIR, 'metrics')
HISTORY_DIR = os.path.normpath(
    os.path.join(TOF_DIR, '..', '..', 'old', 'tof_calibration_history')
)


def discover_candidates():
    """List of (label, path): active calibration first, then archived files."""
    out = []
    active = os.path.join(METRICS_DIR, 'default_params.txt')
    if os.path.exists(active):
        out.append(('active', active))

    for path in sorted(glob.glob(os.path.join(HISTORY_DIR, '**', '*.txt'),
                                 recursive=True)):
        rel = os.path.relpath(path, HISTORY_DIR)
        parts = rel.replace('\\', '/').split('/')
        label = parts[0].replace('attempt_', '') if len(parts) > 1 else parts[0]
        if len(parts) > 1:
            stem = os.path.splitext(parts[-1])[0]
            stem = stem.replace('calibrated_params_', '').replace('_default_params', '')
            label = f'{label}/{stem}' if stem else label
        out.append((label, path))
    return out


def main():
    columns = []
    for label, path in discover_candidates():
        if os.path.exists(path):
            columns.append((label, read_params_file(path)))

    if not columns:
        print('No parameter files found in', METRICS_DIR, 'or', HISTORY_DIR)
        sys.exit(1)

    rows = ALL_PARAM_NAMES + ['objective']
    label_width = max(len(name) for name in rows)
    col_width = max(16, max(len(lbl) for lbl, _ in columns) + 2)

    header = f'{"":<{label_width}}  ' + '  '.join(f'{lbl:>{col_width}}' for lbl, _ in columns)
    print(header)
    print('-' * len(header))

    for name in rows:
        cells = []
        for _, params in columns:
            v = params.get(name, None)
            cells.append(f'{"--":>{col_width}}' if v is None else f'{v:>{col_width}.6g}')
        print(f'{name:<{label_width}}  ' + '  '.join(cells))

    print()
    print('Highlights:')
    for name in ['concentration_scale_0p2', 'concentration_scale_0p8']:
        parts = []
        for lbl, params in columns:
            v = params.get(name, None)
            parts.append(f'{lbl}={v:.4g}' if v is not None else f'{lbl}=--')
        print(f'  {name}: ' + ', '.join(parts))


if __name__ == '__main__':
    main()
