#!/bin/bash
# Recovery fine-tuning after iterative pruning: start from the 40k pruned models (run_40k/),
# keep their masks fixed, train LoRA 1 epoch on 20k LaMini samples NOT seen during pruning
# (shuffled with the same seed, offset 41000). LoRA snapshots every 50 steps.
# Weights -> run_finetune/ft_rXX (results/, logs/ were moved to run_finetune/ afterwards).
# Eval: perplexity + zero-shot for every snapshot, 5-shot MMLU for the final model.
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate loraprune
cd ~/LoRAPrune
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

zs() {  # TAG LORA_DIR : perplexity + zero-shot only
    python inference.py --base_model huggyllama/llama-7b --lora_weights "$2" --cutoff_len 2048 \
        --lora_r 8 --lora_alpha 16 --lora_dropout 0.0 \
        --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
        > logs/$1_ppl.log 2>&1 && grep -E "ppl" logs/$1_ppl.log > results/$1_ppl.txt
    python eval_lmeval.py --lora_weights "$2" --num_fewshot 0 --output results/$1_zeroshot.json \
        > logs/$1_zeroshot.log 2>&1
    echo "[$(date)] [$1] ppl+zero-shot done: $(tr '\n' ' ' < results/$1_ppl.txt)"
}

run() {  # RATIO TAG GPU DELAY
    sleep "$4"
    export CUDA_VISIBLE_DEVICES=$3
    python prune.py --base_model huggyllama/llama-7b --data_path MBZUAI/LaMini-instruction \
        --output_dir "run_finetune/ft_$2" --init_lora "run_40k/outputs_$2/adapter_model.bin" \
        --nsamples 20000 --data_offset 41000 --lora_save_steps 50 \
        --lr_scheduler_type cosine --batch_size 128 --micro_batch_size 2 --num_epochs 1 \
        --learning_rate 1e-4 --cutoff_len 512 --val_set_size 1000 \
        --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
        --lora_target_modules '[q_proj, k_proj, v_proj, o_proj, gate_proj,up_proj, down_proj]' \
        --train_on_inputs --group_by_length --ratio "$1" --prune_metric lora --prune_freq 10 \
        > "logs/ft_$2.log" 2>&1
    if grep -q "Traceback" "logs/ft_$2.log" || [ ! -s "run_finetune/ft_$2/adapter_model.bin" ]; then
        echo "[$(date)] [ft_$2] FINE-TUNING FAILED - see logs/ft_$2.log"; return 1
    fi
    echo "[$(date)] [ft_$2] fine-tuning finished OK"
    for s in 50 100; do zs "ft_$2_s$s" "run_finetune/ft_$2/step-$s"; done
    zs "ft_$2_final" "run_finetune/ft_$2"
    python eval_lmeval.py --lora_weights "run_finetune/ft_$2" --tasks mmlu --num_fewshot 5 \
        --output results/ft_$2_final_mmlu.json > logs/ft_$2_final_mmlu.log 2>&1
    echo "[$(date)] [ft_$2] done"
}

run 0.5 r50 0 0 &
run 0.2 r20 1 60 &
run 0.3 r30 2 120 &
wait
echo "[$(date)] === ALL DONE ==="
