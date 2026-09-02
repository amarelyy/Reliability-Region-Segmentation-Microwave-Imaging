"""
metrics_result/fig_feature_rationale.py

Generates a compact diagram explaining why only peak dominance and
boundary risk are used in the reliability score.

Output: metrics_result/fig_feature_rationale.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "metrics_result" / "fig_feature_rationale.png"


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Title
    ax.text(6, 6.6, "Reliability Score Feature Selection Rationale",
            fontsize=14, fontweight="bold", ha="center", va="center")

    # === LEFT: All Available Metrics ===
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.3, 3.2), 3.4, 3.0, boxstyle="round,pad=0.1",
        facecolor="#e3f2fd", edgecolor="#1565c0", linewidth=2
    ))
    ax.text(2.0, 5.9, "All Available\nPipeline Metrics", fontsize=11,
            fontweight="bold", ha="center", va="center", color="#1565c0")

    all_metrics = [
        ("Peak Dominance", "#2ca02c", True),
        ("SCR (dB)", "#ff9800", False),
        ("CNR", "#ff9800", False),
        ("CF at Peak", "#ff9800", False),
        ("SMR (dB)", "#ff9800", False),
        ("Blob Compactness", "#ff9800", False),
        ("Boundary Risk", "#d32f2f", True),
    ]

    y_start = 5.4
    for i, (name, color, selected) in enumerate(all_metrics):
        y = y_start - i * 0.35
        marker = "★" if selected else "•"
        weight = "bold" if selected else "normal"
        ax.text(0.6, y, f"{marker} {name}", fontsize=9, fontweight=weight,
                va="center", color=color)

    # === MIDDLE: Correlation Arrows ===
    # Group 1: Peak Dominance cluster (collinear)
    ax.annotate("", xy=(4.5, 4.8), xytext=(3.7, 4.8),
                arrowprops=dict(arrowstyle="->", color="#ff9800", lw=2))
    ax.text(4.1, 5.1, "Collinear\n(r > 0.85)", fontsize=8, ha="center",
            va="center", color="#ff9800", fontstyle="italic")

    # Group 2: Boundary Risk (orthogonal)
    ax.annotate("", xy=(4.5, 3.6), xytext=(3.7, 3.6),
                arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=2))
    ax.text(4.1, 3.9, "Orthogonal\n(r < 0.2)", fontsize=8, ha="center",
            va="center", color="#d32f2f", fontstyle="italic")

    # === RIGHT TOP: Selected Features ===
    ax.add_patch(mpatches.FancyBboxPatch(
        (4.8, 4.2), 3.4, 2.0, boxstyle="round,pad=0.1",
        facecolor="#e8f5e9", edgecolor="#2e7d32", linewidth=2
    ))
    ax.text(6.5, 5.9, "Selected for\nReliability Score", fontsize=11,
            fontweight="bold", ha="center", va="center", color="#2e7d32")
    ax.text(5.1, 5.3, "★ Peak Dominance", fontsize=10, fontweight="bold",
            va="center", color="#2ca02c")
    ax.text(5.1, 4.8, "   = SCR + exclusion zones", fontsize=8,
            va="center", color="#555555", fontstyle="italic")
    ax.text(5.1, 4.4, "★ Boundary Risk", fontsize=10, fontweight="bold",
            va="center", color="#d32f2f")

    # === RIGHT BOTTOM: Excluded Features ===
    ax.add_patch(mpatches.FancyBboxPatch(
        (4.8, 1.5), 3.4, 2.2, boxstyle="round,pad=0.1",
        facecolor="#fbe9e7", edgecolor="#c62828", linewidth=1.5, linestyle="--"
    ))
    ax.text(6.5, 3.4, "Excluded", fontsize=11,
            fontweight="bold", ha="center", va="center", color="#c62828")

    excluded_reasons = [
        ("SCR, CNR, CF, SMR", "Redundant with Peak Dominance"),
        ("Blob Compactness", "Weak correlation with quality"),
    ]
    for i, (feat, reason) in enumerate(excluded_reasons):
        y = 2.9 - i * 0.5
        ax.text(5.1, y, f"✗ {feat}", fontsize=9, va="center", color="#c62828")
        ax.text(5.1, y - 0.2, f"   {reason}", fontsize=8, va="center",
                color="#777777", fontstyle="italic")

    # === FAR RIGHT: Final Formula ===
    ax.add_patch(mpatches.FancyBboxPatch(
        (8.8, 2.8), 3.0, 3.4, boxstyle="round,pad=0.1",
        facecolor="#fff3e0", edgecolor="#e65100", linewidth=2
    ))
    ax.text(10.3, 5.9, "Final Reliability\nScore", fontsize=11,
            fontweight="bold", ha="center", va="center", color="#e65100")

    formula_lines = [
        "Reliability =",
        "  min(1, Dom/5.0)",
        "  × (1 − Risk)",
        "",
        "Dom ∈ [0, ∞)",
        "Risk ∈ [0, 1]",
        "Score ∈ [0, 1]",
    ]
    for i, line in enumerate(formula_lines):
        weight = "bold" if i < 3 else "normal"
        size = 10 if i < 3 else 9
        ax.text(9.1, 5.3 - i * 0.35, line, fontsize=size, fontweight=weight,
                va="center", color="#333333", family="monospace")

    # Arrow from selected to formula
    ax.annotate("", xy=(8.8, 4.5), xytext=(8.2, 4.5),
                arrowprops=dict(arrowstyle="->", color="#e65100", lw=2.5))

    # === BOTTOM: Key Insight Box ===
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.3, 0.3), 11.4, 1.0, boxstyle="round,pad=0.1",
        facecolor="#f3e5f5", edgecolor="#6a1b9a", linewidth=2
    ))
    ax.text(6.0, 0.8,
            "Key Insight: Peak Dominance and Boundary Risk capture ORTHOGONAL quality dimensions.\n"
            "A scan can have high dominance (clean peak) AND high risk (strong ring artifact) simultaneously.\n"
            "All other metrics are collinear with one or the other — adding them provides zero new information.",
            fontsize=9, ha="center", va="center", color="#4a148c", fontweight="bold")

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()