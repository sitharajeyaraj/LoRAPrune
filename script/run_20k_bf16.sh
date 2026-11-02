#!/bin/bash
# 20k-sample rerun (bf16) with cosine LR: 20/30/50% in parallel on GPUs 1/2/0, each followed by
# full evaluation; unpruned baseline results are reused. Summary -> results/SUMMARY.md
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate loraprune
cd ~/LoRAPrune
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NSAMPLES=20000 LR_SCHED=cosine RUN_NAME="20k LaMini samples, cosine LR, bf16"

run() {  # RATIO TAG GPU DELAY
    sleep "$4"  # stagger dataset preprocessing
    script/prune_ratio.sh "$1" "outputs_$2" "$3" > "logs/prune_$2.log" 2>&1
    if grep -q "Traceback" "logs/prune_$2.log" || [ ! -s "outputs_$2/adapter_model.bin" ]; then
        echo "[$(date)] [$2] PRUNING FAILED - see logs/prune_$2.log"; return 1
    fi
    echo "[$(date)] [$2] pruning finished OK"
    script/eval_all.sh "$2" "outputs_$2" "$3"
}

run 0.5 r50 0 0 &
run 0.2 r20 1 60 &
run 0.3 r30 2 120 &
wait
python script/summarize.py
echo "[$(date)] === ALL DONE === see results/SUMMARY.md"
