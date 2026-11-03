"""Estimate the adipose-shell radius and centre at the antenna plane from the signal itself
(ground-truth-free): first strong echo of each bistatic pair in the adipose-only reference scan
(empty-chamber subtracted). For a circle of radius rho centred at c, the specular point of a pair
at angles (phi, phi+60) lies near the bisector psi = phi+30 deg, and the round trip is
2*sqrt(Ra^2 + rho^2 - 2*Ra*rho*cos(30 deg)). Per position, rho_k is solved from the echo time;
then rho_k ~= R + cx*cos(psi_k) + cy*sin(psi_k) gives R and the centre (first order in |c|/R).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data_loading import load_all_data
from src import signal_processing as sp, physics as ph

C = 3e8
GATE_NS, MAX_NS = 0.95, 3.0   # after the direct Tx->Rx path (~0.85 ns at r = 0.25 m)


def rho_from_tau(tau, Ra, half=np.deg2rad(30)):
    # (tau*C/2)^2 = Ra^2 + rho^2 - 2 Ra rho cos(half)  ->  smaller root (front face of the shell)
    d2 = (tau * C / 2) ** 2
    b = Ra * np.cos(half)
    disc = b ** 2 - (Ra ** 2 - d2)
    return b - np.sqrt(np.maximum(disc, 0))


def estimate(fd_adi, fd_emp, ant_rad_cm):
    td = np.abs(sp.to_time_domain(fd_adi - fd_emp))
    t = sp.get_time_axis(td.shape[0])
    win = (t >= GATE_NS * 1e-9) & (t <= MAX_NS * 1e-9)
    k = np.argmax(td[win], axis=0)
    tau = t[win][k]
    g = ph.get_antenna_geometry(ant_rad_cm * 10)
    Ra = g["ant_rad_m"]
    psi = np.arctan2(g["ant_y"] + g["ant_y_b"], g["ant_x"] + g["ant_x_b"])
    rho = rho_from_tau(tau, Ra) * 1000  # mm
    A = np.c_[np.ones_like(psi), np.cos(psi), np.sin(psi)]
    good = np.ones(len(rho), bool)
    for _ in range(3):  # robust refit: drop positions > 3 MAD from the fit
        coef, *_ = np.linalg.lstsq(A[good], rho[good], rcond=None)
        res = rho - A @ coef
        mad = np.median(np.abs(res[good])) + 1e-9
        good = np.abs(res) < 3 * 1.4826 * mad
    return dict(R_est_mm=coef[0], cx_est_mm=coef[1], cy_est_mm=coef[2],
                rho_resid_mm=float(np.std(res[good])), n_used=int(good.sum()))


if __name__ == "__main__":
    d = load_all_data()
    tm, s21, idm = d["tumor_model"], d["s21"], d["id_to_original_idx"]
    refs = tm.drop_duplicates("adi_ref_id")
    rows = []
    for _, r in refs.iterrows():
        e = estimate(s21[idm[int(r.adi_ref_id)]], s21[idm[int(r.emp_ref_id)]], r.ant_rad)
        rows.append(dict(adi_ref_id=int(r.adi_ref_id), phant_id=r.phant_id,
                         adi=r.phant_id.split("F")[0], ant_rad=r.ant_rad, R_table_mm=r.breast_radius_mm,
                         adi_x_mm=r.adi_x * 10, adi_y_mm=r.adi_y * 10, **e))
    df = pd.DataFrame(rows)
    out = REPO / "results_audit/shell_estimates.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df.groupby("adi").agg(n=("R_est_mm", "size"), R_table=("R_table_mm", "first"),
                                R_est_med=("R_est_mm", "median"), R_est_sd=("R_est_mm", "std"),
                                resid=("rho_resid_mm", "median")).round(1))
    m = (df.adi_x_mm != 0) | (df.adi_y_mm != 0)
    print("\ncentre check on refs with non-zero adi_x/adi_y (n=%d):" % m.sum())
    print("  corr(cx_est, adi_x) = %.2f, corr(cy_est, adi_y) = %.2f" %
          (np.corrcoef(df.cx_est_mm[m], df.adi_x_mm[m])[0, 1], np.corrcoef(df.cy_est_mm[m], df.adi_y_mm[m])[0, 1]))
    print(df.loc[m, ["phant_id", "adi_x_mm", "adi_y_mm", "cx_est_mm", "cy_est_mm"]].round(1).head(15).to_string(index=False))
