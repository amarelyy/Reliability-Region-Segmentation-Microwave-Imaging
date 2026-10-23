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
        figsize=(7, 7),
        facecolor='white'
    )

    # ========================================================
    # LARGER + MORE CENTERED MATRIX
    # ========================================================
    #
    # Increased matrix size substantially.
    # The matrix now occupies more of the canvas,
    # reducing unnecessary white space.
    #
    ax = fig.add_axes([
        0.27,   # left
        0.25,   # bottom
        0.48,   # width
        0.48    # height
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
    # CELL TEXT
    # ========================================================
    conditions = [
        ["Mod CC\nSafe G", "High CC\nSafe G",
         "Crit CC\nCrit G", "Crit CC\nCrit G"],

        ["Mod CC\nSafe G", "Mod CC\nSafe G",
         "High CC\nMod G", "High CC\nMod G"],

        ["Low CC\nSafe G", "Mod CC\nSafe G",
         "Mod CC\nMod G", "Mod CC\nMod G"],

        ["Low CC\nSafe G", "Low CC\nSafe G",
         "Low CC\nSafe G", "Low CC\nSafe G"]
    ]

    for i in range(4):
        for j in range(4):

            txt_color = 'white' if risk_data[i, j] >= 2 else 'black'

            ax.text(
                j + 0.5,
                i + 0.5,
                conditions[i][j],
                ha='center',
                va='center',
                fontsize=6.5,
                fontweight='bold',
                color=txt_color,
                linespacing=0.9
            )

    # ========================================================
    # X-AXIS LABELS
    # ========================================================
    x_labels = [
        'Low\n(CC<0.3, G>0.8)',
        'Med-Low\n(CC<0.6, G>0.5)',
        'Med-High\n(CC>0.6, G<0.5)',
        'High\n(CC>0.6, G<0.25)'
    ]

    ax.set_xticks(np.arange(4) + 0.5)

    ax.set_xticklabels(
        x_labels,
        fontsize=6,
        rotation=30,
        ha='right',
        va='top'
    )

    ax.tick_params(
        axis='x',
        pad=2,
        length=3
    )

    # ========================================================
    # Y-AXIS LABELS
    # ========================================================
    y_labels = [
        'High\n(D<0.4)',
        'Med-High\n(D<0.7)',
        'Med-Low\n(D>0.7)',
        'Low\n(D>0.9)'
    ]

    ax.set_yticks(np.arange(4) + 0.5)

    ax.set_yticklabels(
        y_labels,
        fontsize=6,
        va='center'
    )

    ax.tick_params(
        axis='y',
        pad=3,
        length=3
    )

    # ========================================================
    # AXIS TITLES
    # ========================================================
    ax.set_xlabel(
        'Likelihood of geometric artifact (CC & G)',
        fontsize=7.5,
        labelpad=7
    )

    ax.set_ylabel(
        'Impact on detection accuracy ($D_{norm}$)',
        fontsize=7.5,
        labelpad=8
    )

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
    # TITLE — CLOSER TO MATRIX
    # ========================================================
    fig.text(
        0.50,
        0.88,
        'RRS-MBI Clinical Safety Risk Matrix',
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold'
    )

    # ========================================================
    # SAVE
    # ========================================================
    out_path = OUTPUT_DIR / "rrs_clinical_risk_matrix.png"

    plt.savefig(
        out_path,
        dpi=600,
        facecolor='white',
        edgecolor='none'
    )

    plt.close(fig)

    print(f"[Saved] {out_path}")
    print("Output: 4200 × 4200 pixels")
    print("Aspect ratio: 1:1")


if __name__ == "__main__":
    create_tight_square_risk_matrix()