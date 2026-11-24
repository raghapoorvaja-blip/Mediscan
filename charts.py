"""
charts.py
---------
Small Matplotlib helpers that draw the three charts used in the app.
They return Figure objects; Streamlit displays them with st.pyplot().
(We use matplotlib's Figure class directly instead of pyplot to avoid global state.)
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Wedge

TEAL = "#0E9AA7"
TEAL_LIGHT = "#BFE6EA"
INK = "#12343B"
STATUS_COLOURS = {"high": "#12A37F", "moderate": "#F2A93B", "low": "#E5534B"}


def probability_chart(labels, probs, predicted_index):
    """Horizontal bars showing the model's probability for every class."""
    fig = Figure(figsize=(5.5, 2.4))
    ax = fig.subplots()
    percentages = [p * 100 for p in probs]
    colours = [TEAL if i == predicted_index else TEAL_LIGHT for i in range(len(labels))]
    positions = np.arange(len(labels))
    ax.barh(positions, percentages, color=colours)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", color=INK, fontsize=9)
    for pos, value in zip(positions, percentages):
        ax.text(min(value + 1.5, 88), pos, f"{value:.1f}%", va="center", fontsize=9, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def similarity_chart(names, similarities):
    """Bar chart of heatmap similarity for each image variation, with the threshold lines."""
    fig = Figure(figsize=(7, 3))
    ax = fig.subplots()
    colours = [STATUS_COLOURS["high"] if s >= 80 else STATUS_COLOURS["moderate"] if s >= 60
               else STATUS_COLOURS["low"] for s in similarities]
    positions = np.arange(len(names))
    ax.bar(positions, similarities, color=colours)
    ax.axhline(80, color=STATUS_COLOURS["high"], linestyle="--", linewidth=1)
    ax.axhline(60, color=STATUS_COLOURS["moderate"], linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8, color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Heatmap similarity (%)", fontsize=9, color=INK)
    for pos, value in zip(positions, similarities):
        ax.text(pos, value + 1.5, f"{value:.0f}", ha="center", fontsize=8, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def score_gauge(score, status_key):
    """Half-circle gauge for the Evidence Stability Score (red / amber / green zones)."""
    fig = Figure(figsize=(4.2, 2.5))
    ax = fig.subplots()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.25, 1.15)

    # 0 -> left end (180 degrees), 100 -> right end (0 degrees)
    def angle(value):
        return 180 - 1.8 * value

    zones = [(0, 60, "#F6B8B4"), (60, 80, "#FBDDA6"), (80, 100, "#A5E2D2")]
    for low, high, colour in zones:
        ax.add_patch(Wedge((0, 0), 1.0, angle(high), angle(low), width=0.28, color=colour))

    # Filled part showing the actual score
    ax.add_patch(Wedge((0, 0), 1.0, angle(score), 180, width=0.28, color=STATUS_COLOURS[status_key], alpha=0.95))

    # Needle
    theta = np.radians(angle(score))
    ax.plot([0, 0.68 * np.cos(theta)], [0, 0.68 * np.sin(theta)], color=INK, linewidth=2)
    ax.add_patch(Wedge((0, 0), 0.05, 0, 360, color=INK))

    ax.text(0, -0.2, f"{score:.0f} / 100", ha="center", va="center", fontsize=18, fontweight="bold", color=INK)
    ax.text(-1.0, 0.0, "0", ha="center", va="top", fontsize=8, color=INK)
    ax.text(1.0, 0.0, "100", ha="center", va="top", fontsize=8, color=INK)
    fig.tight_layout()
    return fig
