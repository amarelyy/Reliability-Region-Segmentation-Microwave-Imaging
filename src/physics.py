import numpy as np

C_LIGHT = 3e8
EPSILON_AIR = 1.0006
V_AIR = C_LIGHT / np.sqrt(EPSILON_AIR)   # ~299.9 mm/ns

N_ANT = 72
SEPARATION_DEG = 60.0

def compute_effective_velocity(fat_frac, fib_frac, eps_fat=7.0, eps_fib=45.0):
    """Hitung kecepatan efektif berdasarkan komposisi phantom."""
    eps_eff = fat_frac * eps_fat + fib_frac * eps_fib
    v_eff = C_LIGHT / np.sqrt(eps_eff)
    return v_eff, eps_eff

def get_corrected_ant_radius_m(raw_rad_mm):
    """Rodriguez-Herrera (2016) antenna radius correction. Input mm, output metres."""
    return (0.97 * (raw_rad_mm - 0.106) + 0.148) / 1000.0

def get_antenna_geometry(ant_rad_mm, n_ant=N_ANT, separation_deg=SEPARATION_DEG, apply_correction=True):
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

def snellius_bistatic_delay_precise(ant_x, ant_y, ant_x_b, ant_y_b, grid_x_m, grid_y_m, 
                                    breast_radius_m, v_tissue, v_air=C_LIGHT):
    """
    Menghitung delay bistatic (Tx->Grid->Rx) dengan koreksi Snellius presisi.
    Menggunakan line-circle intersection analitik untuk menentukan panjang jalur di Udara vs Tissue.
    """
    n_ant = len(ant_x)
    n_pix = len(grid_x_m)
    delay_grid = np.zeros((n_ant, n_pix))
    
    tx_pos = np.stack([ant_x, ant_y], axis=1) # (72, 2)
    rx_pos = np.stack([ant_x_b, ant_y_b], axis=1) # (72, 2)
    grid_pos = np.stack([grid_x_m, grid_y_m], axis=1) # (N_pix, 2)
    
    R = breast_radius_m
    
    for i in range(n_ant):
        tx = tx_pos[i]
        rx = rx_pos[i]
        
        # --- LEG 1: Tx -> Grid Point ---
        vec_tx = grid_pos - tx # (N_pix, 2)
        dist_tx_total = np.linalg.norm(vec_tx, axis=1)
        
        # Line-Circle Intersection for Tx leg
        a1 = np.sum(vec_tx**2, axis=1)
        b1 = 2 * np.sum(tx * vec_tx, axis=1)
        c1 = np.sum(tx**2) - R**2
        disc1 = b1**2 - 4*a1*c1
        
        # FIX: Initialize as ARRAYS, not floats
        dist_air_tx = dist_tx_total.copy()
        dist_tissue_tx = np.zeros_like(dist_tx_total)
        
        valid1 = disc1 >= 0
        if np.any(valid1):
            sqrt_disc1 = np.sqrt(np.maximum(disc1[valid1], 0))
            t1 = (-b1[valid1] - sqrt_disc1) / (2*a1[valid1])
            t2 = (-b1[valid1] + sqrt_disc1) / (2*a1[valid1])
            
            # Entry is the first intersection
            t_entry = np.where(t1 > 1e-6, t1, t2)
            
            is_inside = t_entry < 1.0
            idx_inside = np.where(valid1)[0][is_inside]
            
            if len(idx_inside) > 0:
                t_val = t_entry[is_inside]
                dist_air_tx[idx_inside] = t_val * dist_tx_total[idx_inside]
                dist_tissue_tx[idx_inside] = (1 - t_val) * dist_tx_total[idx_inside]
                
        # --- LEG 2: Grid Point -> Rx ---
        vec_rx = rx - grid_pos # (N_pix, 2)
        dist_rx_total = np.linalg.norm(vec_rx, axis=1)
        
        a2 = np.sum(vec_rx**2, axis=1)
        b2 = 2 * np.sum(grid_pos * vec_rx, axis=1)
        c2 = np.sum(grid_pos**2) - R**2
        disc2 = b2**2 - 4*a2*c2
        
        # FIX: Initialize as ARRAYS, not floats
        dist_air_rx = dist_rx_total.copy()
        dist_tissue_rx = np.zeros_like(dist_rx_total)
        
        valid2 = disc2 >= 0
        if np.any(valid2):
            sqrt_disc2 = np.sqrt(np.maximum(disc2[valid2], 0))
            t1_2 = (-b2[valid2] - sqrt_disc2) / (2*a2[valid2])
            t2_2 = (-b2[valid2] + sqrt_disc2) / (2*a2[valid2])
            
            # Exit point is the first intersection from grid towards rx
            t_exit = np.where(t1_2 > 1e-6, t1_2, t2_2)
            
            is_inside_2 = t_exit < 1.0
            idx_inside_2 = np.where(valid2)[0][is_inside_2]
            
            if len(idx_inside_2) > 0:
                t_val_2 = t_exit[is_inside_2]
                dist_tissue_rx[idx_inside_2] = t_val_2 * dist_rx_total[idx_inside_2]
                dist_air_rx[idx_inside_2] = (1 - t_val_2) * dist_rx_total[idx_inside_2]

        # --- TOTAL DELAY ---
        # Time = (Dist_Air / V_AIR) + (Dist_Tissue / V_TISSUE)
        delay_grid[i] = (dist_air_tx + dist_air_rx) / v_air + \
                        (dist_tissue_tx + dist_tissue_rx) / v_tissue

    return delay_grid