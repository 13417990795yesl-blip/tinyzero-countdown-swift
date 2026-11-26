# 复现与数据索引

## 生成 Countdown 题目

[gen_countdown.py](../gen_countdown.py) 用固定随机种子生成 20,000 条训练题和 256 条评估题。原脚本入口的输出位置是实验机器上的绝对路径。可在仓库根目录运行下面的代码，把同一批题生成到本地 `data/`。每行包含 `messages`、`target` 和 `nums`。

```python
import json
from pathlib import Path
from gen_countdown import gen

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
for split, count, seed in (("train", 20000, 42), ("test", 256, 7)):
    path = data_dir / f"countdown_{split}.jsonl"
    with path.open("w", encoding="utf-8") as out:
        for row in gen(count, seed):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
```

## 独立测试与评测脚本

使用固定种子 20261126 生成候选题，跳过训练集、原验证集和已接受测试题中的重复项。查重键由排序后保留重复数字的列表和目标值组成。原脚本生成的 `data/countdown_test.jsonl` 是历史验证集。以下命令在仓库根目录运行。

```bash
python -m scripts.build_holdout \
  --train data/countdown_train.jsonl \
  --validation data/countdown_test.jsonl \
  --output /tmp/countdown_holdout_256_seed20261126.jsonl

python -m scripts.evaluate_holdout \
  --model /path/to/exported-hf-model \
  --dataset data/countdown_holdout_256_seed20261126.jsonl \
  --output-prefix /tmp/countdown_eval
```

将重新生成的文件与仓库中的[独立测试集](../data/countdown_holdout_256_seed20261126.jsonl)核对即可验证复现结果。评测脚本在已有 ms-swift、Transformers、PyTorch 和 CUDA 的环境运行，读取可直接由 Transformers 加载的模型目录。GRPO 和 PPO 分别使用已合并 LoRA 和已导出为 Hugging Face 格式的最佳验证检查点。模型、数据和输出位置均可通过命令行指定。[检查点与结果](holdout.md)

## 早期 1.5B GRPO 基线

[run_1p5b_grpo.sh](../scripts/run_1p5b_grpo.sh) 保存了第一版 200 步基线的训练配置。运行前按实际机器替换脚本中的工作目录、Python 环境、模型、数据和插件路径，并按 GPU 数量同步调整 `CUDA_VISIBLE_DEVICES` 与 `NPROC_PER_NODE`。

| 项目 | 历史配置 |
|---|---|
| 模型 | Qwen2.5-1.5B-Instruct，全参数更新 |
| 环境 | ms-swift 4.6.1，torch 2.15（CUDA 13.2），Transformers 5.22.2，TRL 0.29.1，额外依赖 msgspec |
| 生成 | Transformers 原生生成，每题 8 条，最多 1024 token |
| 硬件 | 6 张 RTX 4090 |
| 优化 | 学习率 `1e-6`，KL 系数 `0.001`，200 步，约 1 小时 |

该基线使用 [Countdown 奖励函数](../countdown_plugin.py)。无完整答案给 0，有完整答案但表达式无效或结果错误给 0.1，结果正确给 1.0。模型可从 [ModelScope 的 Qwen2.5-1.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct) 获取。准备好模型和数据后执行 `bash scripts/run_1p5b_grpo.sh`，可以用 `tensorboard --logdir output` 查看训练曲线。

最早的 200 步运行，前后各 25 步的平均奖励为 0.117 和 0.123，同组奖励完全一致的比例由 0.52 升到 0.62。平均回答长度由 139 变为 144 token，平均最长回答为 889 和 907 token，梯度范数由 1.98 降到 1.38，KL 约由 0.001 升到 0.002。完整指标见 [200 步日志 CSV](../results/metrics_1p5b_grpo.csv)，可用 [TensorBoard 导出脚本](../scripts/export_metrics.py) 从日志生成。

后续的 1.5B v2 在同一评估口径下取得 10.9%，初始模型为 3.1%。这两个检查点的缺完整答案率分别为 0.4% 和 20.7%，达到长度上限的比例为 0.4% 和 0.8%，平均长度为 50 和 190 token。3B 阶段的配置和大模型结果见 [实验报告](experiments.md)。

## 图表数据与重画方式

| 文件 | 内容 |
|---|---|
| [comparison_checkpoints.csv](../results/comparison_checkpoints.csv) | 3B 检查点统一评估、题目覆盖和累计强化学习耗时。v3 末步记录只含训练覆盖，评估字段为空。 |
| [grpo_instability.csv](../results/grpo_instability.csv) | v2/v3 第 491–800 步的训练采样答对数、KL、长度、达到上限比例和裁剪前梯度范数。前 9 步用于绘制第 500 步的移动平均。 |
| [v2_v3_curves.csv](../results/v2_v3_curves.csv) | v2/v3 长程训练平均奖励和 KL，包含后期退化区间。 |
| [ppo_training.csv](../results/ppo_training.csv) | PPO 390 步的训练奖励、回答长度与达到上限比例。 |
| [ppo_checkpoint_eval.csv](../results/ppo_checkpoint_eval.csv) | PPO 专项评估图的输入，含 GRPO v2 第 800 步参考点。 [PNG](../results/ppo_checkpoint_eval.png) · [PDF](../results/ppo_checkpoint_eval.pdf) |

下面的命令从公开 CSV 重画图表。需要 Matplotlib，中文对比图还需要 Noto Sans CJK SC 或 Microsoft YaHei 等中文字体。

```shell
python scripts/plot_comparison.py
python scripts/plot_ppo_training.py
python scripts/plot_ppo_results.py
```

仓库公开了题目生成、早期 1.5B 运行脚本、指标数据和绘图代码。3B 运行参数记录在 [实验报告](experiments.md#训练方式)，完整启动脚本、权重和训练检查点存放在原实验环境中。
