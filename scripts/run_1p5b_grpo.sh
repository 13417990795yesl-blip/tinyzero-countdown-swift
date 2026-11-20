#!/bin/bash
cd /data1/yesiliang/tinyzero_swift
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NPROC_PER_NODE=6
export PYTHONPATH=/data1/yesiliang/tinyzero_swift:$PYTHONPATH
/data1/yesiliang/ms-swift/venv/bin/swift rlhf \
    --rlhf_type grpo \
    --model /data1/yesiliang/tinyzero_swift/models/Qwen2.5-1.5B-Instruct \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --dataset /data1/yesiliang/tinyzero_swift/data/countdown_train.jsonl \
    --external_plugins /data1/yesiliang/tinyzero_swift/countdown_plugin.py \
    --reward_funcs countdown \
    --use_vllm false \
    --num_generations 8 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --beta 0.001 \
    --temperature 1.0 \
    --max_steps 200 \
    --save_steps 50 \
    --logging_steps 1 \
    --log_completions true \
    --report_to tensorboard \
    --output_dir /data1/yesiliang/tinyzero_swift/output/qwen2.5-1.5b-countdown-grpo
