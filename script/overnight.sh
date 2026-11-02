#!/bin/bash
# Overnight queue: GPU0 = finish 50% run -> eval r50 -> eval unpruned base
#                  GPU1 = prune 20% -> eval r20 ;  GPU2 = prune 30% -> eval r30
# then build results/SUMMARY.md
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate loraprune
cd ~/LoRAPrune
P50=353921

check_prune() {  # LOG OUTDIR TAG
    if grep -q "Traceback" "$1" || [ ! -s "$2/adapter_model.bin" ]; then
        echo "[$(date)] [$3] PRUNING FAILED - see $1"; return 1
    fi
    echo "[$(date)] [$3] pruning finished OK"
}

(
    while kill -0 $P50 2>/dev/null; do sleep 30; done
    check_prune prune.log outputs_dir r50 && script/eval_all.sh r50 outputs_dir 0
    script/eval_all.sh base "" 0
) &
(
    script/prune_ratio.sh 0.2 outputs_r20 1 > logs/prune_r20.log 2>&1
    check_prune logs/prune_r20.log outputs_r20 r20 && script/eval_all.sh r20 outputs_r20 1
) &
(
    sleep 60  # stagger dataset preprocessing
    script/prune_ratio.sh 0.3 outputs_r30 2 > logs/prune_r30.log 2>&1
    check_prune logs/prune_r30.log outputs_r30 r30 && script/eval_all.sh r30 outputs_r30 2
) &
wait
python script/summarize.py
echo "[$(date)] === ALL DONE === see results/SUMMARY.md"
