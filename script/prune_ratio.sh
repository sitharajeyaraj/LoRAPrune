#!/bin/bash
# Usage: prune_ratio.sh RATIO OUTPUT_DIR GPU   (same settings as script/prune.sh, different ratio)
RATIO=$1; OUT=$2; GPU=$3
CUDA_VISIBLE_DEVICES=$GPU python prune.py \
    --base_model "huggyllama/llama-7b" \
    --data_path 'MBZUAI/LaMini-instruction' \
    --output_dir "$OUT" \
    --nsamples "${NSAMPLES:-20000}" \
    --lr_scheduler_type "${LR_SCHED:-linear}" \
    --batch_size 128 \
    --micro_batch_size 2 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 1000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj, k_proj, v_proj, o_proj, gate_proj,up_proj, down_proj]' \
    --train_on_inputs \
    --group_by_length \
    --ratio "$RATIO" \
    --prune_metric 'lora' \
    --prune_freq 10
