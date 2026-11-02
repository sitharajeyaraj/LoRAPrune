#!/bin/bash
# Usage: eval_all.sh TAG LORA_DIR GPU   (LORA_DIR="" evaluates the unpruned base model)
# Runs WikiText2/PTB perplexity, 7 zero-shot tasks and 5-shot MMLU; results go to results/TAG_*
TAG=$1; LORA=$2; GPU=$3
cd ~/LoRAPrune
export CUDA_VISIBLE_DEVICES=$GPU
echo "[$(date)] [$TAG] perplexity"
python inference.py --base_model huggyllama/llama-7b --lora_weights "$LORA" --cutoff_len 2048 \
    --lora_r 8 --lora_alpha 16 --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    > logs/${TAG}_ppl.log 2>&1 && grep -E "ppl" logs/${TAG}_ppl.log | tee results/${TAG}_ppl.txt \
    || echo "[$TAG] perplexity FAILED - see logs/${TAG}_ppl.log"
echo "[$(date)] [$TAG] zero-shot"
python eval_lmeval.py --lora_weights "$LORA" --num_fewshot 0 --output results/${TAG}_zeroshot.json \
    > logs/${TAG}_zeroshot.log 2>&1 || echo "[$TAG] zero-shot FAILED - see logs/${TAG}_zeroshot.log"
echo "[$(date)] [$TAG] 5-shot MMLU"
python eval_lmeval.py --lora_weights "$LORA" --tasks mmlu --num_fewshot 5 --output results/${TAG}_mmlu.json \
    > logs/${TAG}_mmlu.log 2>&1 || echo "[$TAG] MMLU FAILED - see logs/${TAG}_mmlu.log"
echo "[$(date)] [$TAG] done"
