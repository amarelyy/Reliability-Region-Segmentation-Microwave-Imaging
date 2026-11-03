import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm

# ============================================================
# IEEE-STYLE SETTINGS
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.linewidth': 0.8,
})

OUTPUT_DIR = Path("metrics_image")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_tight_square_risk_matrix():

    # ========================================================
    # RISK DATA
    # ========================================================
    risk_data = np.array([
        [1, 2, 3, 3],
        [1, 1, 2, 2],
        [0, 1, 1, 1],
        [0, 0, 0, 0]
    ])

    # ========================================================
    # RISK COLORS
    # ========================================================
    colors = [
        '#2ECC40',   # Low
        '#FFDC00',   # Moderate
        '#FF851B',   # High
        '#FF4136'    # Critical
    ]

    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # ========================================================
    # TRUE 1:1 CANVAS
    # ========================================================
    fig = plt.figure(
        figsize=(6.5, 6.5),
        facecolor='white'
    )

    # ========================================================
    # TIGHTER MATRIX LAYOUT
    # ========================================================
    ax = fig.add_axes([
        0.15,   # left
        0.12,   # bottom
        0.72,   # width
        0.72    # height
    ])

    # ========================================================
    # DRAW MATRIX
    # ========================================================
    ax.pcolormesh(
        risk_data,
        cmap=cmap,
        norm=norm,
        edgecolors='#BDBDBD',
        linewidth=0.6
    )

    # ========================================================
    # CELL TEXT (Tanpa markdown, 2 baris: status + parameter)
    # ========================================================
    # Baris 1: Status (akan di-render bold terpisah)
    status_text = [
        ["Mod CC, Safe G", "High CC, Safe G", "Crit CC, Crit G", "Crit CC, Crit G"],
        ["Mod CC, Safe G", "Mod CC, Safe G", "High CC, Mod G", "High CC, Mod G"],
        ["Low CC, Safe G", "Mod CC, Safe G", "Mod CC, Mod G", "Mod CC, Mod G"],
        ["Low CC, Safe G", "Low CC, Safe G", "Low CC, Safe G", "Low CC, Safe G"]
    ]

    # Baris 2: Parameter (italic)
    param_text = [
        ["(D<0.4, CC<0.3, G>0.8)", "(D<0.4, CC<0.6, G>0.5)", "(D<0.4, CC>0.6, G<0.5)", "(D<0.4, CC>0.6, G<0.25)"],
        ["(D<0.7, CC<0.3, G>0.8)", "(D<0.7, CC<0.6, G>0.5)", "(D<0.7, CC>0.6, G<0.5)", "(D<0.7, CC>0.6, G<0.25)"],
        ["(D>0.7, CC<0.3, G>0.8)", "(D>0.7, CC<0.6, G>0.5)", "(D>0.7, CC>0.6, G<0.5)", "(D>0.7, CC>0.6, G<0.25)"],
        ["(D>0.9, CC<0.3, G>0.8)", "(D>0.9, CC<0.6, G>0.5)", "(D>0.9, CC>0.6, G<0.5)", "(D>0.9, CC>0.6, G<0.25)"]
    ]

    for i in range(4):
        for j in range(4):
            txt_color = 'white' if risk_data[i, j] >= 2 else 'black'

            # Baris 1: Status (Bold) - sedikit di atas center
            ax.text(
                j + 0.5,
                i + 0.58,
                status_text[i][j],
                ha='center',
                va='center',
                fontsize=5.2,
                fontweight='bold',
                color=txt_color
            )

            # Baris 2: Parameter (Italic) - sedikit di bawah center
            ax.text(
                j + 0.5,
                i + 0.42,
                param_text[i][j],
                ha='center',
                va='center',
                fontsize=4.5,
                fontstyle='italic',
                color=txt_color
            )

    # ========================================================
    # X-AXIS LABELS
    # ========================================================
    x_labels = ['Low', 'Med-Low', 'Med-High', 'High']
    ax.set_xticks(np.arange(4) + 0.5)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=0, ha='center', va='top')
    ax.tick_params(axis='x', pad=4, length=3)

    # ========================================================
    # Y-AXIS LABELS
    # ========================================================
    y_labels = ['High', 'Med-High', 'Med-Low', 'Low']
    ax.set_yticks(np.arange(4) + 0.5)
    ax.set_yticklabels(y_labels, fontsize=7, va='center')
    ax.tick_params(axis='y', pad=4, length=3)

    # ========================================================
    # AXIS TITLES
    # ========================================================
    ax.set_xlabel('Likelihood of geometric artifact (CC & G)', fontsize=8, labelpad=6)
    ax.set_ylabel('Impact on detection accuracy ($D_{norm}$)', fontsize=8, labelpad=6)

    # ========================================================
    # PERFECTLY SQUARE MATRIX
    # ========================================================
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)

    # ========================================================
    # CLEAN IEEE FRAME
    # ========================================================
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # ========================================================
    # TITLE (Diturunkan posisinya agar lebih jauh dari matriks)
    # ========================================================
    fig.suptitle(
        'RRS-MBI Clinical Safety Risk Matrix',
        fontsize=12,
        fontweight='bold',
        y=0.88   # Diturunkan dari 0.96 ke 0.88 → jarak lebih jauh dari matriks
    )

    # ========================================================
    # SAVE
    # ========================================================
    out_path = OUTPUT_DIR / "rrs_clinical_risk_matrix.png"
    plt.savefig(out_path, dpi=600, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close(fig)

    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    create_tight_square_risk_matrix()