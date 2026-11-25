"""Draw the comparison figures as PNG images from the repository CSVs."""

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt


PPO = "#00668D"
GRPO2 = "#A33D63"
GRPO3 = "#9A6209"
SFT = "#34785D"
INK = "#172836"


def config():
    names = set(font_manager.get_font_names())
    candidates = (
        "Noto Sans CJK SC", "Microsoft YaHei", "Microsoft YaHei UI",
        "Noto Sans CJK JP", "SimHei", "Droid Sans Fallback",
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
    return ((data["ppo"], PPO, "o", "PPO"), (data["v2"], GRPO2, "s", "SFT + GRPO v2"),
            (data["v3"], GRPO3, "^", "SFT + GRPO v3"))


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
    for rows, color, marker, name in series(data):
        start = data["base"] if name == "PPO" else data["sft"]
        x = [0] + [r["secs"] / 3600 for r in rows]
        y = [pct(start, "correct")] + [pct(r, "correct") for r in rows]
        for i in range(1, len(x)):
            ax.plot(x[i-1:i+1], y[i-1:i+1], color=color, linewidth=2.0,
                    linestyle="--" if i == 1 else "-", alpha=0.7 if i == 1 else 1)
        ax.scatter(x[1:], y[1:], c=color, marker=marker, s=45, label=name, zorder=3)
    ax.scatter([0, 0], [pct(data["base"], "correct"), pct(data["sft"], "correct")],
               c=[PPO, SFT], s=45, zorder=3)
    label(ax, 0, pct(data["base"], "correct"), "3B Instruct  23.4%", PPO, 8, -4)
    label(ax, 0, pct(data["sft"], "correct"), "SFT  30.9%", SFT, 32, -10)
    for bucket, step, color, dx, dy, align in (
        ("ppo", 78, PPO, 5, 9, "left"), ("ppo", 390, PPO, -5, 10, "right"),
        ("v2", 800, GRPO2, 9, 5, "left"), ("v3", 550, GRPO3, -5, 13, "right"),
    ):
        r = next(row for row in data[bucket] if row["step"] == step)
        value = pct(r, "correct")
        label(ax, r["secs"]/3600, value, f"{step} 步 · {value:.1f}%", color, dx, dy, align)
    ax.set(xlabel="累计强化学习训练耗时（小时）", ylabel="严格正确率（%）",
           xlim=(-0.15, 7.8), ylim=(16, 96))
    style_axes(ax)
    ax.legend(loc="lower right", ncols=3, fontsize=10, frameon=False)
    save(fig, folder, "comparison_accuracy_time")


def coverage(data, folder):
    fig, ax = plt.subplots(figsize=(9.7, 4.5), layout="constrained")
    interesting = [
        (data["ppo"][0], PPO, "o", "PPO · 78 步", -9, 9, "right"),
        (data["ppo"][-1], PPO, "o", "PPO · 390 步", -9, 9, "right"),
        (next(r for r in data["v2"] if r["step"] == 800), GRPO2, "s", "v2 · 800 步", 8, 5, "left"),
        (data["v3"][-1], GRPO3, "^", "v3 · 550 步", 8, -16, "left"),
        (data["v3_full"], GRPO3, "^", "v3 · 全轮 1,666 步", -9, 7, "right"),
    ]
    ppo = data["ppo"]
    ax.plot([r["distinct"]/1000 for r in ppo], [r["responses"]/1000 for r in ppo],
            color=PPO, linewidth=1.8, zorder=2)
    ax.scatter([r["distinct"]/1000 for r in ppo[1:-1]],
               [r["responses"]/1000 for r in ppo[1:-1]], s=30, color=PPO, zorder=3)
    for row, color, marker, text, dx, dy, align in interesting:
        x, y = row["distinct"]/1000, row["responses"]/1000
        ax.scatter(x, y, s=63, marker=marker,
                   facecolors="white" if row is data["v3_full"] else color,
                   edgecolors=color, linewidths=1.7, zorder=4)
        label(ax, x, y, text, color, dx, dy, align, fontsize=11)
    ax.set(xlabel="累计见过的不同题目（千道）", ylabel="累计自采样回答（千条）",
           xlim=(0, 21.3), ylim=(0, 177))
    style_axes(ax)
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
