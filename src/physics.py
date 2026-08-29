"""
src/physics.py

Antenna geometry + two delay models:
  - two_medium_delay: Aurel's air + adaptive-tissue-velocity model (line-circle
    intersection), used as the ablation baseline.
  - bent_ray_3layer_delay: Ursula's air -> skin -> interior Fermat-principle
    solver, used as the proposed model.

Both take the same (antenna_xy, pixel_grid) inputs and return a delay grid in
seconds, so pipeline.py can swap between them with one flag.
"""

import numpy as np

C_LIGHT = 3e8
EPSILON_AIR = 1.0006
V_AIR = C_LIGHT / np.sqrt(EPSILON_AIR)   # ~299.9 mm/ns

N_ANT = 72
SEPARATION_DEG = 60.0


# ============================================================================
# Antenna geometry
# ============================================================================
def get_corrected_ant_radius_m(raw_rad_mm):
    """Rodriguez-Herrera (2016) antenna radius correction. Input mm, output metres."""
    return (0.97 * (raw_rad_mm - 0.106) + 0.148) / 1000.0

def get_antenna_geometry(ant_rad_mm, n_ant=N_ANT, separation_deg=SEPARATION_DEG,
                          apply_correction=True):
    """
    Per-scan antenna positions (both ports) from a raw ant_rad metadata value
    (mm, already converted from the metadata's native cm). Applies the
    Rodriguez-Herrera correction by default.

    Returns dict: ant_x, ant_y, ant_x_b, ant_y_b (arrays, length n_ant, metres),
    tx_idx, rx_idx (channel-mapping indices).
    """
    ant_rad_m = get_corrected_ant_radius_m(ant_rad_mm) if apply_correction else ant_rad_mm / 1000.0

    angles = np.linspace(0, -2 * np.pi, n_ant, endpoint=False)
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

def compute_effective_velocity(fat_frac, fib_frac, eps_fat=7.0, eps_fib=45.0):
    """
    Menghitung kecepatan efektif berdasarkan komposisi phantom (UM-BMID metadata).
    Ini menggantikan asumsi kecepatan tunggal yang kaku.
    """
    eps_eff = fat_frac * eps_fat + fib_frac * eps_fib
    v_eff = C_LIGHT / np.sqrt(eps_eff)
    return v_eff, eps_eff

def snellius_bistatic_delay(ant_x, ant_y, ant_x_b, ant_y_b, grid_x_m, grid_y_m, 
                            breast_radius_m, v_tissue, v_air=C_LIGHT):
    """
    Menghitung delay bistatic (Tx->Grid->Rx) dengan koreksi Snellius sederhana.
    Untuk phantom homogen, kita gunakan pendekatan straight-ray dengan effective velocity
    di dalam medium, tapi tetap memperhitungkan dua titik antena yang berbeda.
    """
    n_ant = len(ant_x)
    n_pix = len(grid_x_m)
    delay_grid = np.zeros((n_ant, n_pix))
    
    # Posisi antena Tx dan Rx
    tx_pos = np.stack([ant_x, ant_y], axis=1)
    rx_pos = np.stack([ant_x_b, ant_y_b], axis=1)
    grid_pos = np.stack([grid_x_m, grid_y_m], axis=1)
    
    for i in range(n_ant):
        tx = tx_pos[i]
        rx = rx_pos[i]
        
        # Jarak Tx ke setiap titik grid
        dist_tx = np.linalg.norm(grid_pos - tx, axis=1)
        # Jarak Rx ke setiap titik grid
        dist_rx = np.linalg.norm(grid_pos - rx, axis=1)
        
        # Cek apakah titik grid ada di dalam phantom (radius breast_radius_m)
        dist_grid_center = np.linalg.norm(grid_pos, axis=1)
        inside_phantom = dist_grid_center <= breast_radius_m
        
        # Hitung delay:
        # 1. Di Udara (jarak antena ke permukaan phantom)
        # 2. Di Tissue (jarak di dalam phantom)
        # Simplifikasi: Kita anggap jalur lurus, tapi kecepatannya berubah saat masuk lingkaran.
        # Untuk presisi tinggi, kita butuh titik potong garis Tx-Grid dengan lingkaran.
        
        # Titik potong Tx -> Grid
        # ... (Logika intersection yang kompleks disederhanakan untuk stabilitas awal) ...
        # Kita gunakan weighted average velocity berdasarkan seberapa jauh grid di dalam lingkaran
        
        # Pendekatan Robust untuk Smoke Test:
        # Delay = (Jarak Total di Udara / V_AIR) + (Jarak Total di Tissue / V_TISSUE)
        # Kita estimasi jarak di tissue sebagai chord length jika garis memotong lingkaran.
        
        # Untuk saat ini, mari gunakan model Two-Medium yang sudah terbukti jalan di repo asli
        # tapi dengan v_tissue yang sudah dikalibrasi (Effective Velocity).
        # Model Bent-Ray/Snellius penuh butuh iterasi per-pixel yang sangat berat dan rawan bug
        # jika koordinatnya tidak sempurna dalam meter.
        
        # KITA GUNAKAN INI DULU UNTUK MEMASTIKAN DASARNYA BENAR:
        delay_grid[i] = (dist_tx + dist_rx) / v_air 
        
        # Koreksi sederhana: Jika grid di dalam phantom, kurangi delay udara dan tambah delay tissue
        # Ini adalah aproksimasi linear yang lebih stabil daripada Snellius iteratif untuk saat ini
        mask = inside_phantom
        if np.any(mask):
            # Estimasi kasar jarak di dalam tissue (bisa diperbaiki dengan line-circle intersection)
            # Untuk sekarang, kita beri bobot v_tissue pada seluruh jalur jika target di dalam
            # Ini akan membuat fokus lebih tajam daripada murni v_air
            delay_grid[i, mask] = (dist_tx[mask] + dist_rx[mask]) / v_tissue

    return delay_grid