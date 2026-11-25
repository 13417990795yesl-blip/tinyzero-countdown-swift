"""Plot strict accuracy and length-limit rates for the PPO checkpoints."""

import csv

from ppo_plot_style import GRPO_COLOR, PPO_COLOR, RESULTS, plt, save_figure


def main():
    with (RESULTS / "ppo_checkpoint_eval.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    ppo = [row for row in rows if row["model"] == "PPO"]
    grpo, = (row for row in rows if row["model"] == "SFT+GRPO")
    steps = [int(row["step"]) for row in ppo]
    metrics = [
        ("strict_correct", "Strict accuracy (%)", (70, 92)),
        ("at_length_limit", "At length limit (%)", (0, 23)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.55), layout="constrained")
    for ax, (column, label, limits) in zip(axes, metrics):
        values = [100 * int(row[column]) / int(row["n"]) for row in ppo]
        reference = 100 * int(grpo[column]) / int(grpo["n"])
        ax.plot(steps, values, color=PPO_COLOR, marker="o", linewidth=2, label="PPO")
        ax.axhline(
            reference,
            color=GRPO_COLOR,
            linestyle="--",
            linewidth=1.5,
            label=f"SFT+GRPO (step {grpo['step']})",
        )
        for step, value in zip(steps, values):
            ax.annotate(
                f"{value:.1f}%",
                (step, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                color=PPO_COLOR,
                fontsize=9,
            )
        ax.annotate(
            f"{reference:.1f}%",
            (1, reference),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            color=GRPO_COLOR,
            fontsize=9,
        )
        ax.set_ylabel(label)
        ax.set_xlabel("PPO checkpoint step")
        ax.set_xticks(steps)
        ax.set_xlim(steps[0] - 29, steps[-1] + 29)
        ax.set_ylim(*limits)
        ax.grid(axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
    save_figure(fig, "ppo_checkpoint_eval")


if __name__ == "__main__":
    main()
