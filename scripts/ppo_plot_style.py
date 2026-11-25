"""Shared figure settings for the Countdown PPO results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

RESULTS = Path(__file__).resolve().parents[1] / "results"
PPO_COLOR = "#0072B2"
GRPO_COLOR = "#D55E00"
RESUME_COLOR = "#595959"

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.color": "#dedede",
        "grid.linewidth": 0.7,
        "pdf.fonttype": 42,
    }
)


def save_figure(figure, name):
    figure.savefig(RESULTS / f"{name}.pdf")
    figure.savefig(RESULTS / f"{name}.png", dpi=300)
    plt.close(figure)
