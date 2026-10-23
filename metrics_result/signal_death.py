import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- IEEE Formal Styling ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0

def load_scr_data():
    try:
        # Ganti dengan path file metrik SCR Anda yang sebenarnya
        # df = pd.read_csv('metrics_result/scr_by_radius.csv')
        # return df[['radial_distance_mm', 'scr_db']]
        raise FileNotFoundError("Mocking data for demonstration.")
    except Exception:
        np.random.seed(42)
        n = 150
        radial_dist = np.random.uniform(15, 50, n)
        # Model: +10 dB di dekat pusat, turun ke -24 dB di tepi (r=50mm)
        scr_db = 10 - ((radial_dist - 15) / 35) * 34 + np.random.normal(0, 2.5, n)
        return pd.DataFrame({'radial_distance_mm': radial_dist, 'scr_db': scr_db})

def main():
    os.makedirs("metrics_image", exist_ok=True)
    df = load_scr_data()
    
    fig, ax = plt.subplots(figsize=(7, 5)) # Aspect ratio lebar untuk kurva tren
    
    # 1. Scatter plot data points
    ax.scatter(df['radial_distance_mm'], df['scr_db'], color='#1f77b4', alpha=0.6, 
               s=25, edgecolor='white', linewidth=0.5, label='Scan Measurements')
    
    # 2. Trendline (Ordinary Least Squares)
    coeffs = np.polyfit(df['radial_distance_mm'], df['scr_db'], 1)
    trendline = np.poly1d(coeffs)
    x_trend = np.linspace(df['radial_distance_mm'].min(), df['radial_distance_mm'].max(), 100)
    ax.plot(x_trend, trendline(x_trend), color='red', linewidth=2.0, 
            label=f'Degradation Trendline (Slope: {coeffs[0]:.2f} dB/mm)')
    
    # 3. Detectability threshold at 0 dB
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Detectability Threshold (0 dB)')
    
    ax.set_title('Signal Death Curve: SCR vs. Tumor Radial Distance')
    ax.set_xlabel('Radial Distance of Tumor Center (mm)')
    ax.set_ylabel('Signal-to-Clutter Ratio (SCR) [dB]')
    
    ax.set_ylim(-30, 15)
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Hilangkan spine kanan dan atas untuk tampilan IEEE yang lebih bersih
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('metrics_image/signal_death_curve.png', dpi=600, bbox_inches='tight')
    print("Saved: metrics_image/signal_death_curve.png")
    plt.close()

if __name__ == "__main__":
    main()