"""Draw the comparison figures as PNG images from the repository CSVs."""

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt


PPO = "#963626"
GRPO2 = "#193B65"
GRPO3 = "#007A76"
INK = "#172836"


def config():
    names = set(font_manager.get_font_names())
    candidates = (
        "Noto Sans CJK SC", "Microsoft YaHei", "Microsoft YaHei UI",
        "Noto Sans CJK JP", "SimHei", "Droid Sans Fallback",
        "Noto Serif CJK SC", "Noto Serif CJK JP",
    )
    font = next((name for name in candidates if name in names), None)
    if font is None:
        raise SystemExit("Install a Chinese font such as Noto Sans CJK SC to render the charts.")
    plt.rcParams.update({
        "font.family": font, "font.size": 12, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "axes.axisbelow": True, "axes.labelpad": 9,
        "grid.color": "#dde4e8", "grid.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.unicode_minus": False,
    })


def pct(row, key):
    return row[key] * 100 / row["n"]


def series(data):
    return ((data["v2"], GRPO2, "o", "-", "SFT + GRPO v2"),
            (data["v3"], GRPO3, "D", "-.", "SFT + GRPO v3"),
            (data["ppo"], PPO, "s", "--", "PPO"))


def style_axes(ax):
    ax.tick_params(length=0, pad=6)
    ax.grid(axis="y")


def label(ax, x, y, text, color, dx=5, dy=9, ha="left", fontsize=10):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, fontsize=fontsize, ha=ha)


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=165, facecolor="white")
    plt.close(fig)


def time_accuracy(data, folder):
    fig, ax = plt.subplots(figsize=(9.7, 4.8), layout="constrained")
    start = pct(data["base"], "correct")
    for rows, color, marker, linestyle, name in series(data):
        x = [0] + [r["secs"] / 3600 for r in rows]
        y = [start] + [pct(r, "correct") for r in rows]
        ax.plot(x, y, color=color, marker=marker, markersize=5.5,
                linewidth=2, linestyle=linestyle, label=name)
    ax.scatter([0], [start], c=INK, s=39, zorder=4)
    label(ax, 0, start, f"3B Instruct  {start:.1f}%", INK, 8, -4)
    for bucket, step, color, dx, dy, align in (
        ("ppo", 78, PPO, 5, 9, "left"), ("ppo", 390, PPO, -5, 10, "right"),
        ("v2", 800, GRPO2, 9, 5, "left"), ("v3", 550, GRPO3, -5, 13, "right"),
    ):
        r = next(row for row in data[bucket] if row["step"] == step)
        value = pct(r, "correct")
        label(ax, r["secs"]/3600, value, f"{step} 步 · {value:.1f}%", color, dx, dy, align)
    ax.set(xlabel="累计强化学习训练耗时（小时）", ylabel="验证集严格正确率（%）",
           xlim=(-0.15, 7.8), ylim=(16, 96))
    style_axes(ax)
    ax.legend(loc="lower right", ncols=3, fontsize=10, frameon=False)
    save(fig, folder, "comparison_accuracy_time")


def coverage(data, folder):
    fig, ax = plt.subplots(figsize=(9.7, 4.5), layout="constrained")
    for rows, color, marker, linestyle, name in series(data):
        points = [*rows, data["v3_full"]] if name == "SFT + GRPO v3" else rows
        x = [0] + [r["responses"]/10000 for r in points]
        y = [0] + [r["distinct"]/10000 for r in points]
        ax.plot(x, y, color=color, marker=marker, markersize=5.5,
                linewidth=2, linestyle=linestyle, label=name)
    ax.scatter([0], [0], c=INK, s=39, zorder=4)
    ax.annotate("3B Instruct", (0, 0), xytext=(8, -2), textcoords="offset points",
                color=INK, fontsize=10, va="top")
    v3_end = data["v3_full"]
    label(ax, v3_end["responses"]/10000, v3_end["distinct"]/10000,
          "v3 · 1,666 步", GRPO3, 0, 8, "right")
    ax.set(xlabel="累计自采样回答（万条）", ylabel="累计覆盖的不同题目（万道）",
           xlim=(-0.15, 16.7), ylim=(-0.04, 2.16))
    style_axes(ax)
    ax.legend(loc="lower right", ncols=3, frameon=False, fontsize=10)
    save(fig, folder, "comparison_coverage")


def instability(data, folder):
    fig, axes = plt.subplots(3, 1, figsize=(9.7, 6.65), sharex=True, layout="constrained")
    panels = (("correct", "答对比例（%）", 100/96), ("kl", "平均 KL", 1),
              ("length", "平均回答长度（token）", 1))
    for ax, (field, title, scale) in zip(axes, panels):
        for key, color, legend in (("v2", GRPO2, "GRPO v2"), ("v3", GRPO3, "GRPO v3")):
            rows = data["curves"][key]
            steps = list(range(500, 801))
            y = [scale * mean(rows[str(i)][field] for i in range(s-9, s+1)) for s in steps]
            ax.plot(steps, y, linewidth=2.25, label=legend, color=color)
        for step, color in ((633, GRPO3), (741, GRPO2)):
            ax.axvline(step, color=color, alpha=0.48, linestyle="--", linewidth=1.1)
        ax.set_ylabel(title)
        style_axes(ax)
    axes[0].set(ylim=(0, 101))
    axes[1].set_ylim(bottom=0)
    axes[-1].set(xlabel="GRPO 优化步数", xlim=(500, 800), xticks=range(500, 801, 50))
    for step, color in ((633, GRPO3), (741, GRPO2)):
        axes[0].annotate(f"{step} 步", xy=(step, 93), xytext=(5, 0), textcoords="offset points",
                         color=color, fontsize=10, va="center")
    axes[0].legend(loc="lower left", ncols=2, frameon=False, fontsize=10)
    save(fig, folder, "grpo_instability")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_comparison(root):
    data = {"ppo": [], "v2": [], "v3": [], "curves": {"v2": {}, "v3": {}}}
    columns = {
        "step": "step", "rl_wall_seconds": "secs", "unique_questions": "distinct",
        "sampled_responses": "responses", "n": "n", "correct": "correct",
    }
    for row in read_csv(root / "comparison_checkpoints.csv"):
        run = row["run"]
        entry = {key: int(row[column]) for column, key in columns.items() if row[column]}
        if run in ("base", "sft"):
            data[run] = entry
        elif run == "v3" and not row["n"]:
            data["v3_full"] = entry
        else:
            data[run].append(entry)

    for row in read_csv(root / "grpo_instability.csv"):
        data["curves"][row["run"]][row["step"]] = {
            "correct": int(row["correct_out_of_96"]),
            "kl": float(row["kl"]),
            "length": float(row["mean_response_tokens"]),
        }
    return data


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    parser = argparse.ArgumentParser(description="Recreate the cross-run Countdown charts.")
    parser.add_argument("--output-dir", type=Path, default=root)
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    data = load_comparison(root)
    config()
    for plot in (time_accuracy, coverage, instability):
        plot(data, output)


if __name__ == "__main__":
    main()
