"""Plot PPO training reward and response lengths from the exported log metrics."""

import csv
from statistics import mean

from ppo_plot_style import PPO_COLOR, RESUME_COLOR, RESULTS, plt, save_figure

WINDOW = 10


def main():
    with (RESULTS / "ppo_training.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    steps = [int(row["step"]) for row in rows]
    epoch_ends = [
        int(row["step"])
        for row, next_row in zip(rows, rows[1:])
        if row["epoch"] != next_row["epoch"]
    ]
    resume_step = epoch_ends[0]
    metrics = [
        ("mean_reward", "Mean reward", 1),
        ("mean_response_tokens", "Mean response length\n(tokens)", 1),
        ("at_length_limit_rate", "At length limit (%)", 100),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for ax, (column, label, scale) in zip(axes, metrics):
        values = [float(row[column]) * scale for row in rows]
        smoothed = [mean(values[end - WINDOW : end]) for end in range(WINDOW, len(values) + 1)]
        ax.plot(steps, values, color=PPO_COLOR, alpha=0.28, linewidth=0.9, label="Per step")
        ax.plot(
            steps[WINDOW - 1 :],
            smoothed,
            color=PPO_COLOR,
            linewidth=2,
            label=f"{WINDOW}-step trailing mean",
        )
        ax.axvline(
            resume_step,
            color=RESUME_COLOR,
            linewidth=1,
            linestyle="--",
            label=f"Resumed after step {resume_step}",
        )
        ax.set_ylabel(label)
        ax.grid(axis="y")
    axes[-1].set_xlabel("PPO training step")
    axes[-1].set_xticks([0, *epoch_ends, steps[-1]])
    axes[-1].set_xlim(0, steps[-1] + 4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    fig.align_ylabels(axes)
    save_figure(fig, "ppo_training")


if __name__ == "__main__":
    main()
