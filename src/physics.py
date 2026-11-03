import numpy as np

C_LIGHT = 3e8
EPSILON_AIR = 1.0006
V_AIR = C_LIGHT / np.sqrt(EPSILON_AIR)   # ~299.9 mm/ns

N_ANT = 72
SEPARATION_DEG = 60.0
PORT_A_START_DEG = -130.0   # UM-BMID Gen-2 (docs/scan_geometry.md)

def compute_effective_velocity(fat_frac, fib_frac, eps_fat=7.0, eps_fib=45.0):
    """Hitung kecepatan efektif berdasarkan komposisi phantom."""
    eps_eff = fat_frac * eps_fat + fib_frac * eps_fib
    v_eff = C_LIGHT / np.sqrt(eps_eff)
    return v_eff, eps_eff

def get_corrected_ant_radius_m(raw_rad_mm):
    """Rodriguez-Herrera (2016) antenna radius correction. Input mm, output metres."""
    raw_rad_m = raw_rad_mm / 1000.0          # UM-BMID formula is in metres
    return 0.97 * (raw_rad_m - 0.106) + 0.148

def get_antenna_geometry(ant_rad_mm, n_ant=N_ANT, separation_deg=SEPARATION_DEG, apply_correction=True):
    ant_rad_m = get_corrected_ant_radius_m(ant_rad_mm) if apply_correction else ant_rad_mm / 1000.0
    angles = np.deg2rad(PORT_A_START_DEG) + np.linspace(0, -2 * np.pi, n_ant, endpoint=False)
    ant_x = ant_rad_m * np.cos(angles)
    ant_y = ant_rad_m * np.sin(angles)
    offset = np.deg2rad(separation_deg)
    ant_x_b = ant_rad_m * np.cos(angles + offset)
    ant_y_b = ant_rad_m * np.sin(angles + offset)
    sep_steps = int(round(separation_deg / 360.0 * n_ant))
    tx_idx = np.arange(n_ant)
    rx_idx = (np.arange(n_ant) + sep_steps) % n_ant
    return dict(ant_x=ant_x, ant_y=ant_y, ant_x_b=ant_x_b, ant_y_b=ant_y_b,
                tx_idx=tx_idx, rx_idx=rx_idx, ant_rad_m=ant_rad_m)

def _tissue_length(a, b, R):
    """Length of segment a->b (arrays (N,2)) that lies inside the circle of radius R."""
    d = b - a
    A = np.sum(d**2, axis=1)
    B = 2 * np.sum(a * d, axis=1)
    C = np.sum(a**2, axis=1) - R**2
    disc = B**2 - 4 * A * C
    s = np.sqrt(np.maximum(disc, 0.0))
    t1 = np.clip((-B - s) / (2 * A), 0.0, 1.0)
    t2 = np.clip((-B + s) / (2 * A), 0.0, 1.0)
    return np.where(disc > 0, (t2 - t1) * np.sqrt(A), 0.0)


def snellius_bistatic_delay_precise(ant_x, ant_y, ant_x_b, ant_y_b, grid_x_m, grid_y_m,
                                    breast_radius_m, v_tissue, v_air=C_LIGHT):
    """
    Bistatic Tx -> pixel -> Rx delay with a straight-ray two-medium (air/tissue) split.
    Each leg's tissue length is the clipped chord of the segment through the breast circle,
    so it is correct for pixels inside AND outside the breast.
    """
    grid_pos = np.stack([grid_x_m, grid_y_m], axis=1)
    n_pix = len(grid_x_m)
    delay_grid = np.zeros((len(ant_x), n_pix))
    for i in range(len(ant_x)):
        tx = np.broadcast_to([ant_x[i], ant_y[i]], (n_pix, 2))
        rx = np.broadcast_to([ant_x_b[i], ant_y_b[i]], (n_pix, 2))
        L_tx = np.linalg.norm(grid_pos - tx, axis=1)
        L_rx = np.linalg.norm(rx - grid_pos, axis=1)
        t_tx = _tissue_length(tx, grid_pos, breast_radius_m)
        t_rx = _tissue_length(grid_pos, rx, breast_radius_m)
        delay_grid[i] = (L_tx - t_tx + L_rx - t_rx) / v_air + (t_tx + t_rx) / v_tissue
    return delay_grid
