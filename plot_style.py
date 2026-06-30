"""Shared IEEE-style matplotlib defaults, Tol Muted palette, and per-(loading/distance/bin) colour maps."""

import io
import time
import matplotlib
import matplotlib.pyplot as plt


IEEE_SINGLE_COL = 3.5
IEEE_DOUBLE_COL = 7.16


TOL_MUTED = {
    'indigo':     '#332288',
    'cyan':       '#88CCEE',
    'teal':       '#44AA99',
    'green':      '#117733',
    'olive':      '#999933',
    'sand':       '#DDCC77',
    'rose':       '#CC6677',
    'wine':       '#882255',
    'purple':     '#AA4499',
}


LOADING_COLORS = {0.2: TOL_MUTED['indigo'], 0.8: TOL_MUTED['rose']}

DISTANCE_COLORS = {
    10: TOL_MUTED['indigo'],
    20: TOL_MUTED['cyan'],
    30: TOL_MUTED['teal'],
    40: TOL_MUTED['olive'],
    50: TOL_MUTED['rose'],
    60: TOL_MUTED['sand'],
}

TRIAL_COLORS = [TOL_MUTED['cyan'], TOL_MUTED['indigo'], TOL_MUTED['wine']]

BIN_COLORS = [TOL_MUTED['indigo'], TOL_MUTED['cyan'], TOL_MUTED['teal'],
              TOL_MUTED['olive'], TOL_MUTED['rose'], TOL_MUTED['sand']]
BIN_MARKERS = ['o', 's', '^', 'D', 'v', 'p']


PANEL_ORDER  = [10, 40, 20, 50, 30, 60]
PANEL_LABELS = ["10 cm", "40 cm", "20 cm", "50 cm", "30 cm", "60 cm"]

PANEL_YLIMS = {
    10: (0, 12),
    20: (0, 22),
    30: (0, 32),
    40: (0, 43),
    50: (0, 53),
    60: (0, 64),
}


def setup_ieee_style():
    """Configure matplotlib for IEEE figures embedded 1:1 in LaTeX (>=8pt text)."""
    plt.rcParams.update({
        'font.family':       'serif',
        'text.usetex':       False,
        'mathtext.fontset':  'dejavuserif',
        'font.size':         8,
        'axes.labelsize':    9,
        'axes.titlesize':    9,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'legend.fontsize':   7,
        'figure.dpi':        150,
        'axes.grid':         True,
        'grid.linestyle':    '--',
        'grid.linewidth':    0.5,
        'grid.alpha':        0.5,
        'axes.linewidth':    0.5,
        'lines.linewidth':   1.0,
        'axes.spines.top':   True,
        'axes.spines.right': True,
    })


def save_fig(fig, path, dpi=300):
    """Save figure via memory buffer with retry for Windows file locking."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    png_bytes = buf.getvalue()
    buf.close()
    for attempt in range(3):
        try:
            with open(path, 'wb') as f:
                f.write(png_bytes)
            print(f"Saved: {path}")
            return
        except OSError:
            if attempt < 2:
                time.sleep(0.5)
    raise OSError(f"Failed to write {path} after 3 attempts")
