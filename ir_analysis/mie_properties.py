"""Per-bin Mie optical properties for quartz-equivalent dust at 850 nm (miepython preferred, SciPy fallback)."""

from __future__ import annotations

import numpy as np

try:
    import miepython  # type: ignore
except ModuleNotFoundError:
    miepython = None
    from scipy.special import spherical_jn, spherical_yn

WAVELENGTH_UM = 0.850
N_REAL = 1.54
N_IMAG = 0.0
M_COMPLEX = complex(N_REAL, N_IMAG)


def _mie_efficiencies_fallback(m: complex, d_um: float, wavelength_um: float):
    """SciPy Bohren-Huffman Mie efficiencies — fallback for the chamber-bin size range."""
    x = np.pi * float(d_um) / float(wavelength_um)
    if x <= 0.0:
        raise ValueError("size parameter must be positive")

    n_stop = int(np.ceil(x + 4.05 * x ** (1.0 / 3.0) + 2.0))
    n = np.arange(1, n_stop + 1, dtype=float)

    z = complex(m) * x
    j_x = spherical_jn(n.astype(int), x)
    y_x = spherical_yn(n.astype(int), x)
    j_x_p = spherical_jn(n.astype(int), x, derivative=True)
    y_x_p = spherical_yn(n.astype(int), x, derivative=True)

    j_z = spherical_jn(n.astype(int), z)
    j_z_p = spherical_jn(n.astype(int), z, derivative=True)

    psi_x = x * j_x
    psi_x_p = j_x + x * j_x_p
    xi_x = x * (j_x + 1j * y_x)
    xi_x_p = (j_x + 1j * y_x) + x * (j_x_p + 1j * y_x_p)

    psi_z = z * j_z
    psi_z_p = j_z + z * j_z_p

    a_num = m * psi_z * psi_x_p - psi_x * psi_z_p
    a_den = m * psi_z * xi_x_p - xi_x * psi_z_p
    b_num = psi_z * psi_x_p - m * psi_x * psi_z_p
    b_den = psi_z * xi_x_p - m * xi_x * psi_z_p
    a = a_num / a_den
    b = b_num / b_den

    two_n_1 = 2.0 * n + 1.0
    qext = (2.0 / x ** 2) * np.sum(two_n_1 * np.real(a + b))
    qsca = (2.0 / x ** 2) * np.sum(two_n_1 * (np.abs(a) ** 2 + np.abs(b) ** 2))
    qback = (1.0 / x ** 2) * abs(np.sum(two_n_1 * ((-1.0) ** n) * (a - b))) ** 2

    if qsca > 0.0 and len(a) > 1:
        n1 = n[:-1]
        term1 = np.sum(
            n1 * (n1 + 2.0) / (n1 + 1.0) *
            np.real(a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:]))
        )
        term2 = np.sum(
            (2.0 * n + 1.0) / (n * (n + 1.0)) * np.real(a * np.conj(b))
        )
        g = (4.0 / (x ** 2 * qsca)) * (term1 + term2)
    else:
        g = 0.0

    return float(qext), float(qsca), float(qback), float(np.clip(g, -1.0, 1.0))


def mie_efficiencies_single(d_um: float) -> dict:
    """Compute Mie efficiencies for one particle diameter."""
    d = float(d_um)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("d_um must be positive and finite.")

    if miepython is not None:
        qext, qsca, qback, g = miepython.efficiencies(
            M_COMPLEX, d, WAVELENGTH_UM)
    else:
        qext, qsca, qback, g = _mie_efficiencies_fallback(
            M_COMPLEX, d, WAVELENGTH_UM)

    qabs = max(float(qext - qsca), 0.0)
    omega_0 = float(qsca / qext) if qext > 0 else 0.0
    omega_0 = float(np.clip(omega_0, 0.0, 1.0))
    return {
        'Q_ext': float(qext),
        'Q_sca': float(qsca),
        'Q_abs': qabs,
        'Q_back': float(qback),
        'omega_0': omega_0,
        'g': float(g),
    }


def mie_efficiencies_bins(d_rep_um: np.ndarray) -> dict:
    """Compute Mie efficiencies for representative bin diameters."""
    d_rep_um = np.atleast_1d(np.asarray(d_rep_um, dtype=float))
    out = {k: np.zeros(len(d_rep_um), dtype=float)
           for k in ['Q_ext', 'Q_sca', 'Q_abs', 'Q_back', 'omega_0', 'g']}

    for i, d in enumerate(d_rep_um):
        props = mie_efficiencies_single(float(d))
        for k in out:
            out[k][i] = props[k]
    return out
