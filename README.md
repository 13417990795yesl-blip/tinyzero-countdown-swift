# TinyZero Countdown：用 ms-swift 复现 GRPO（Qwen2.5-1.5B-Instruct，小规模版）

本仓库记录一次小规模实验进度：参考 [TinyZero](https://github.com/Jiayi-Pan/TinyZero) 的 Countdown 任务，使用 ms-swift 的 GRPO 训练 Qwen2.5-1.5B-Instruct。这是一个用于课堂汇报的小规模复现，结论不代表 TinyZero 原始结果。

## 目录结构

```
gen_countdown.py               # 按 TinyZero 的 Countdown 格式生成训练/测试数据（固定随机种子）
countdown_plugin.py            # ms-swift 自定义奖励函数（规则奖励，与 TinyZero 的 compute_score 一致）
scripts/run_1p5b_grpo.sh       # 训练启动脚本（swift rlhf --rlhf_type grpo）
scripts/export_metrics.py      # 从 TensorBoard 日志导出指标为 CSV
results/metrics_1p5b_grpo.csv  # 200 步训练的全部指标
```

## 环境

- ms-swift 4.6.1，torch 2.15（CUDA 13.2），transformers 5.22.2，trl 0.29.1
- 6 张 RTX 4090（48G）
- 未使用 vLLM，未使用 deepspeed（生成使用 transformers 原生方式）
- 额外依赖：`msgspec`（缺少时启动会报 `ModuleNotFoundError`）

## 复现步骤

```shell
# 1. 生成数据（输出到 data/，约 2 万条训练样本 + 256 条测试样本）
python gen_countdown.py

# 2. 下载基座模型（ModelScope）
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir models/Qwen2.5-1.5B-Instruct

# 3. 启动训练（路径需要按你的机器修改）
bash scripts/run_1p5b_grpo.sh

# 4. 查看曲线
tensorboard --logdir output
```

运行前请修改 `scripts/run_1p5b_grpo.sh` 中的路径，并根据机器情况调整 `CUDA_VISIBLE_DEVICES`。

## 奖励设计

与 TinyZero 的 `compute_score` 保持一致：

| 情况 | 奖励 |
|---|---|
| 没有 `<answer>...</answer>` 标签 | 0 |
| 有标签，但数字用法不合规或无法计算 | 0.1 |
| 数字合规，但结果不等于目标值 | 0.1 |
| 数字合规，结果等于目标值 | 1.0 |

## 训练设置

| 参数 | 值 |
|---|---|
| 算法 | GRPO |
| 模型 | Qwen2.5-1.5B-Instruct，全参数训练 |
| 每组采样数 `num_generations` | 8 |
| 最大生成长度 | 1024 |
| 学习率 | 1e-6 |
| KL 系数 `beta` | 0.001 |
| 步数 | 200（约 1 小时） |

## 结果（200 步）

数值为前 1/8 与后 1/8 步的平均值。

| 指标 | 前段 | 后段 | 解读 |
|---|---|---|---|
| 奖励均值 `CountdownORM/mean` | 0.117 | 0.123 | 基本未变 |
| 同组奖励全相同的比例 `frac_reward_zero_std` | 0.52 | 0.62 | 上升，无学习信号的组变多 |
| 平均回答长度 | 139 | 144 | 变化很小 |
| 最长回答长度（截断情况） | 889 | 907 | 基本未变 |
| 梯度范数 `grad_norm` | 1.98 | 1.38 | 下降 |
| KL | 0.001 | 0.002 | 模型几乎没有偏离初始状态 |

**结论**：在 200 步、1.5B 模型的设置下，模型没有明显学到 Countdown 的正确解法。奖励 0.1 只表示"格式对但答案错"，奖励均值约 0.12，说明答对的样本比例很低。

## 可能的原因与后续尝试

1. 1.5B Instruct 模型的基础能力较弱，TinyZero 作者的结果以 3B 模型为主；
2. 步数偏少，学习率偏小；
3. 答对率低时，大量题目的 8 条回答奖励完全相同，无法产生优势信号。

计划尝试：Qwen2.5-3B；先用 3 个数字的题目（课程学习）；增加步数；将 `num_generations` 提高到 16。

## 未包含在仓库中的内容

- 模型权重（基座约 3GB）和训练检查点（约 26GB）不在仓库中，需要另外传输或重新训练；
- 数据文件由 `gen_countdown.py` 生成，不单独上传。
