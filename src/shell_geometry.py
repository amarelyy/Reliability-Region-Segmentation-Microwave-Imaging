"""
src/shell_geometry.py

Ground-truth-free estimate of the adipose-shell centre at the antenna plane, from the
adipose-only reference scan (adi_ref_id) minus the empty-chamber scan.

For a circle of radius rho centred at c, the specular point of a bistatic pair at angles
(phi, phi+60 deg) lies near the bisector psi = phi+30 deg, with round trip
2*sqrt(Ra^2 + rho^2 - 2*Ra*rho*cos(30 deg)). rho_k is solved per position from the first
strong echo; rho_k ~= R + cx*cos(psi_k) + cy*sin(psi_k) then gives the centre (first order
in |c|/R).

Only the centre is used downstream. The absolute radius from this fit is biased low by
~15 % (a few tens of ps of system delay shift it; 1 mm of radius ~ 5 ps), so the
breast radius still comes from phantom_database.csv.
"""

import numpy as np

from . import physics as ph
from . import signal_processing as sp

GATE_NS, MAX_NS = 0.95, 3.0   # after the direct Tx->Rx path (~0.85 ns at r = 0.25 m)


def _rho_from_tau(tau, Ra, half=np.deg2rad(30)):
    d2 = (tau * ph.C_LIGHT / 2) ** 2
    b = Ra * np.cos(half)
    return b - np.sqrt(np.maximum(b ** 2 - (Ra ** 2 - d2), 0))


def estimate_shell_circle(fd_adi, fd_emp, ant_rad_cm):
    """Returns dict(cx_mm, cy_mm, r_echo_mm, resid_mm) from one adipose reference scan."""
    td = np.abs(sp.to_time_domain(np.asarray(fd_adi) - np.asarray(fd_emp)))
    t = sp.get_time_axis(td.shape[0])
    win = (t >= GATE_NS * 1e-9) & (t <= MAX_NS * 1e-9)
    tau = t[win][np.argmax(td[win], axis=0)]
    g = ph.get_antenna_geometry(ant_rad_cm * 10)
    psi = np.arctan2(g["ant_y"] + g["ant_y_b"], g["ant_x"] + g["ant_x_b"])
    rho = _rho_from_tau(tau, g["ant_rad_m"]) * 1000
    A = np.c_[np.ones_like(psi), np.cos(psi), np.sin(psi)]
    good = np.ones(len(rho), bool)
    for _ in range(3):  # robust refit: drop positions > 3 MAD from the fit
        coef, *_ = np.linalg.lstsq(A[good], rho[good], rcond=None)
        res = rho - A @ coef
        good = np.abs(res) < 3 * 1.4826 * (np.median(np.abs(res[good])) + 1e-9)
    return dict(cx_mm=float(coef[1]), cy_mm=float(coef[2]), r_echo_mm=float(coef[0]),
                resid_mm=float(np.std(res[good])))
