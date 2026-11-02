# Reproducing LoRAPrune on LLaMA-7B

This branch reproduces **LoRAPrune** (Zhang et al., *LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning*, Findings of ACL 2024) on LLaMA-1 7B. It prunes at 20%, 30% and 50% and evaluates the same benchmarks as the paper's Table 2:

- WikiText-2 and PTB perplexity
- 7 zero-shot tasks
- 5-shot MMLU

It also contains:
- the code fixes needed to run the released code with current libraries,
- a study of how training data size and precision affect the results,
- an extra experiment, recovery fine-tuning after pruning, that the paper does not test.

**Summary of findings**
- The method works, and at **20% pruning we come close to the paper's numbers.**
- At 30% and especially 50%, we land on the numbers of the paper's **own appendix ablations** (Tables 5, 7 and 8). We do not reach its headline Table 2, which reports 11.60 WikiText-2 perplexity at 50%; we get 17.3–18.7.
- The most likely reason is training data size. The released script, run as written, trains on all 2.58M LaMini samples, about 65× more than we used.

---

## Contents
1. [Setup](#1-setup)
2. [Changes to the code](#2-changes-to-the-code)
3. [Experiments](#3-experiments)
4. [Results](#4-results)
5. [Comparison with the paper](#5-comparison-with-the-paper)
6. [Conclusions](#6-conclusions)
7. [How to reproduce](#7-how-to-reproduce)
8. [Files in this branch](#8-files-in-this-branch)

---

## 1. Setup

| | |
|---|---|
| Hardware | 3× NVIDIA A100 80 GB (one run per GPU), 64 CPU cores, 247 GB RAM |
| Environment | conda env from **conda-forge**, Python 3.10 |
| Packages | torch 2.5.1+cu124, transformers 4.46.3, peft 0.13.2, datasets 3.1.0, lm-eval 0.4.7 |
| Model | [`huggyllama/llama-7b`](https://huggingface.co/huggyllama/llama-7b) (LLaMA-1 7B; the paper's `decapoda-research/llama-7b-hf` has been deleted from the Hub) |
| Pruning data | [`MBZUAI/LaMini-instruction`](https://huggingface.co/datasets/MBZUAI/LaMini-instruction) (2,585,615 samples), random subset |

```bash
conda create -n loraprune --override-channels -c conda-forge python=3.10 git
conda activate loraprune
pip install -r requirements.txt
pip install lm-eval==0.4.7
```

**Why LLaMA-1 rather than Llama-3?** The paper only covers LLaMA-1, and the pruning code assumes multi-head attention: it prunes the q, k and v heads together, using a hard-coded head count. Llama-3 uses grouped-query attention (32 query heads, 8 key/value heads), which this code cannot prune correctly.

### Training configuration (all runs)
These are the settings from `script/prune.sh`, which match the paper's §4.1:

| Setting | Value |
|---|---|
| Batch size / micro batch | 128 / 2 (gradient accumulation 64) |
| Epochs | 2 |
| Learning rate | 1e-4, AdamW |
| LoRA | r = 8, α = 16, dropout 0.05, on q, k, v, o, gate, up, down |
| Cutoff length | 512 tokens; `train_on_inputs`, `group_by_length` |
| Pruning | `prune_metric lora`, `prune_freq 10`, cubic sparsity schedule with 10% warm-up and 10% cool-down |
| Validation set | 1000 samples, taken in addition to `nsamples` |
| Never pruned | layers 0–2 and the last layer (`loraprune.utils.freeze`) |

Because those 4 layers are never pruned, a "50%" model has **3.98B** parameters, not 3.4B:

| Ratio | Parameters |
|---|---|
| 0% | 6.76B |
| 20% | 5.65B |
| 30% | 5.09B |
| 50% | 3.98B |

---

## 2. Changes to the code

All changes are against `aim-uofa/LoRAPrune@4da5272`.

### Bug fixes needed to run at all
| File | Problem | Fix |
|---|---|---|
| `prune.py` | `--nsamples` was accepted but never used, so training always used all 2.58M LaMini samples (several days per run). | Shuffle with seed 42 and keep `nsamples + val_set_size` samples. |
| `script/prune.sh` | A trailing `\` on the last line passed a literal `"\"` as an argument, which crashed at step ~32 (`TypeError: str - float`). | Removed. |
| `prune.py` | With peft 0.13, `model.save_pretrained()` does not recognise LoRAPrune's custom LoRA config and writes **empty** adapter files. | Save every `lora_*` tensor (LoRA weights and pruning masks) with `torch.save` to `adapter_model.bin`. |
| `inference.py` | `set_peft_model_state_dict` fails to load those weights with peft 0.13. A missing file only printed a message, and evaluation silently continued on the unpruned model. | Load with `load_state_dict(strict=False)` and check that no LoRA tensors are missing or unexpected. A missing file now raises an error. Also accepts `adapter_model.safetensors`. |
| `loraprune/utils.py` | After heads are physically removed, transformers ≥ 4.40 also reads `num_key_value_heads`, which was still 32, so shapes no longer matched. | Set `num_key_value_heads = num_heads` and `num_key_value_groups = 1`. |
| `script/prune.sh`, `script/evaluate.sh` | The upstream repo had been switched to Llama-3.1-8B-Instruct, and evaluation used 512-token segments. | LLaMA-7B, with 2048-token segments as stated in the paper. |

### fp16 training bug (important)
`loraprune/trainer.py` replaces the Hugging Face training loop with its own, which **skips the fp16 GradScaler logic**. Gradients are never unscaled, and steps with inf/NaN gradients are never skipped. A single fp16 overflow therefore writes NaN into the optimizer state, and from then on every LoRA weight becomes NaN.

This happened in our first 40k run (`run_40k_fp16_failed/`). After step 500, 448 of 672 LoRA tensors became NaN, which gave NaN perplexity and chance-level accuracy.

Fixes:
- `prune.py`: train in **bf16** (`bf16=True` instead of `fp16=True`). bf16 has the same range as fp32, so it cannot overflow this way.
- `loraprune/trainer.py`: skip the optimizer step, and log a warning, when the gradient norm is not finite. None of the bf16 runs triggered this.

Without unscaling, the gradients reaching the importance score are scaled by a constant factor. This does not change *which* channels get pruned: gradients are clipped to norm 1 before both the importance calculation and the optimizer step, and Adam is roughly scale-invariant. Our clean fp16 run (`run_20k/`) is consistent with that; its results are close to the bf16 run on the same data.

### New features
| File | Purpose |
|---|---|
| `prune.py --lr_scheduler_type` | Choose the LR schedule. The paper uses cosine; the code default is linear. |
| `prune.py --init_lora --data_offset --lora_save_steps` | Recovery fine-tuning. These load a pruned model, keep its masks fixed (pruning is never called again), train on samples not seen during pruning, and save snapshots every N steps. |
| `eval_lmeval.py` | Zero-shot and 5-shot MMLU with lm-eval-harness 0.4.7 on the **physically pruned** model. It applies the masks with `prune_from_checkpoint` and passes the model object to `HFLM`. An empty `--lora_weights` evaluates the unpruned model. |
| `script/prune_ratio.sh RATIO OUT GPU` | `prune.sh` with configurable ratio and output folder. `NSAMPLES` and `LR_SCHED` are read from the environment. |
| `script/eval_all.sh TAG LORA_DIR GPU` | Perplexity, then zero-shot, then MMLU, writing `results/TAG_*`. |
| `script/summarize.py` | Builds `results/SUMMARY.md` next to the paper's Table 2. |
| `script/overnight.sh`, `run_40k.sh`, `run_20k_bf16.sh`, `run_finetune.sh` | The exact job queues used for each experiment. Each runs the three ratios in parallel, one per GPU. |

### Evaluation protocol
- **Perplexity:** `inference.py` from the repo, 2048-token segments, WikiText-2 and PTB.
- **Zero-shot:** lm-eval-harness 0.4.7, 0-shot, tasks `boolq piqa hellaswag winogrande arc_easy arc_challenge openbookqa`. The paper does not name its lm-eval version.
  - We report `acc_norm` where a task provides it, and `acc` for BoolQ and WinoGrande. The `SUMMARY.md` files list both.
  - **"Avg (6)"** leaves out BoolQ, matching the paper's Table 2. Its "Average" column is the mean of the 6 tasks it shows, although the caption says seven. The paper averages below are recomputed from Table 2's per-task columns; this reproduces its printed averages at 30% and 50%, but at 20% Table 2 prints 62.70 while its own columns average 60.60.
- **MMLU:** lm-eval-harness `mmlu`, 5-shot, accuracy averaged over all 57 subjects.

---

## 3. Experiments

| Folder | Training samples | Steps | Precision | LR schedule | Notes |
|---|---|---|---|---|---|
| `run_20k/` | 20k | 312 | fp16 | linear | First run. No overflow occurred, so the results are valid. Also holds the unpruned-baseline logs. |
| `run_40k_fp16_failed/` | 40k | 624 | fp16 | cosine | **Failed:** NaN weights after step 500 (see §2). Kept as evidence; results are meaningless. |
| `run_20k_bf16/` | 20k | 312 | bf16 | cosine | Same data as `run_20k`, with the fixed precision and the paper's schedule. |
| `run_40k/` | 40k | 624 | bf16 | cosine | Twice the data; compare with `run_20k_bf16` for the effect of data size alone. |
| `run_finetune/` | +20k unseen | +156 | bf16 | cosine | Starts from `run_40k`; masks fixed; LoRA trained for 1 more epoch. |

Each folder contains `results/` (perplexity `.txt`, lm-eval `.json`, `SUMMARY.md`) and `logs/`. The unpruned baseline (`results/base_*`) was evaluated once and is shared by all runs. Model weights are not in git (≈85 MB per model).

---

## 4. Results

The paper's LoRAPrune results from Table 2 are shown for reference. ↓ = lower is better.

### WikiText-2 perplexity ↓
| Ratio | 20k fp16 | 20k bf16 | 40k bf16 | 40k + fine-tune | **Paper** |
|---|---|---|---|---|---|
| 0% | 5.68 | 5.68 | 5.68 | – | 5.69 |
| 20% | 8.21 | 8.50 | 7.99 | **7.77** | 7.63 |
| 30% | 11.18 | 12.18 | 9.71 | **9.40** | 8.79 |
| 50% | 20.36 | 21.38 | 18.69 | **17.31** | 11.60 |

### PTB perplexity ↓
We compare PTB only between our own runs (see §5.1).
| Ratio | 20k fp16 | 20k bf16 | 40k bf16 | 40k + fine-tune | Paper |
|---|---|---|---|---|---|
| 0% | 29.63 | 29.63 | 29.63 | – | 8.93 |
| 20% | 45.26 | 52.40 | 48.94 | 48.37 | 11.87 |
| 30% | 67.95 | 62.60 | 52.61 | 49.91 | 13.33 |
| 50% | 111.59 | 123.52 | 90.37 | 80.06 | 17.39 |

### Zero-shot average, 6 tasks without BoolQ (same as the paper's Table 2 average)
| Ratio | 20k fp16 | 20k bf16 | 40k bf16 | 40k + fine-tune | **Paper** |
|---|---|---|---|---|---|
| 0% | 64.53 | 64.53 | 64.53 | – | 61.60 |
| 20% | 59.12 | 58.87 | 58.82 | 58.99 | 60.60 |
| 30% | 54.54 | 54.74 | 54.51 | 54.51 | 58.98 |
| 50% | 46.08 | 46.84 | 47.61 | 47.91 | 54.81 |

### Zero-shot average, all 7 tasks
| Ratio | 20k fp16 | 20k bf16 | 40k bf16 | 40k + fine-tune |
|---|---|---|---|---|
| 0% | 66.04 | 66.04 | 66.04 | – |
| 20% | 60.59 | 60.72 | 60.73 | 61.14 |
| 30% | 56.21 | 55.45 | 56.10 | 56.33 |
| 50% | 46.87 | 48.00 | 48.75 | 49.31 |

### 5-shot MMLU (%)
| Ratio | 20k fp16 | 20k bf16 | 40k bf16 | 40k + fine-tune | **Paper** |
|---|---|---|---|---|---|
| 0% | 35.19 | 35.19 | 35.19 | – | 37.10 |
| 20% | 30.40 | 30.77 | 30.10 | 30.29 | 36.81 |
| 30% | 28.54 | 25.12 | 28.56 | 28.84 | 33.60 |
| 50% | 26.53 | 25.09 | 26.15 | 26.56 | 27.84 |

### Per-task zero-shot results: best run (40k + fine-tune) vs paper
| Ratio | | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA |
|---|---|---|---|---|---|---|---|---|
| 0% | ours | 75.11 | 79.16 | 76.20 | 69.85 | 72.73 | 44.62 | 44.60 |
| 0% | paper | (73.18)* | 78.35 | 72.99 | 67.01 | 67.45 | 41.38** | 42.40 |
| 20% | ours | 74.01 | 76.22 | 69.19 | 65.27 | 66.79 | 39.08 | 37.40 |
| 20% | paper | – | 77.48 | 68.64 | 66.77 | 70.20 | 41.89 | 38.60 |
| 30% | ours | 67.28 | 73.72 | 63.10 | 61.72 | 58.80 | 33.70 | 36.00 |
| 30% | paper | – | 77.48 | 66.68 | 62.75 | 69.61 | 41.21 | 36.20 |
| 50% | ours | 57.68 | 68.44 | 50.76 | 55.41 | 50.13 | 30.55 | 32.20 |
| 50% | paper | – | 75.39 | 60.21 | 56.97 | 60.38 | 40.12 | 35.80 |

\* BoolQ is not in Table 2; this value is from Table 5.
\*\* Table 2 repeats the ARC-e value (67.45) in the ARC-c column, an apparent typo; this value is from Table 5.

Full per-task tables, with both `acc` and `acc_norm`, are in each folder's `results/SUMMARY.md` and in `run_finetune/SUMMARY.md`.

### Recovery fine-tuning in detail
Fine-tuning gives most of its gain within the first 50 steps. After step 100 the zero-shot accuracy dips slightly, which is the overfitting the paper warns about in Fig. 6(b).

| Ratio | Steps | WikiText-2 | Avg (7) |
|---|---|---|---|
| 20% | 0 / 50 / 100 / 156 | 7.99 / 7.82 / 7.79 / 7.77 | 60.73 / 61.26 / 61.42 / 61.14 |
| 30% | 0 / 50 / 100 / 156 | 9.71 / 9.43 / 9.41 / 9.40 | 56.10 / 56.40 / 56.53 / 56.33 |
| 50% | 0 / 50 / 100 / 156 | 18.69 / 17.25 / 17.25 / 17.31 | 48.75 / 49.31 / 49.30 / 49.31 |

---

## 5. Comparison with the paper

### 5.1 Unpruned baseline
- **WikiText-2 matches the paper** (5.68 vs 5.69), which confirms that the perplexity code and the model are the same.
- **PTB does not** (29.63 vs 8.93), and none of the variants we tried got close: validation split with lines joined by spaces gave 37.77, the test split gave 46.68, and the tokenizer's `legacy` option made no difference. The paper's PTB protocol is evidently not the one in the released code. **We only compare PTB between our own runs.**
- **Zero-shot:** our baseline is *higher* than the paper's on every task (6-task average 64.5 vs 61.6). This points to a different lm-eval-harness version, prompt format or metric. Comparing each model's drop from its own baseline is therefore fairer than comparing absolute numbers.
- **MMLU:** 35.19 vs 37.10.

### 5.2 Main results (Table 2)

**WikiText-2, as a multiple of the unpruned model's perplexity**
| Ratio | Ours (best) | Paper |
|---|---|---|
| 20% | 1.37× | 1.34× |
| 30% | 1.65× | 1.54× |
| 50% | 3.05× | 2.04× |

**Zero-shot accuracy kept, as a share of the unpruned baseline (6 tasks)**
| Ratio | Ours (best) | Paper |
|---|---|---|
| 20% | 91% | 98% |
| 30% | 84% | 96% |
| 50% | 74% | 89% |

- **At 20% we come close to the paper** (perplexity 7.77 vs 7.63).
- **The gap grows with the pruning ratio.** At 50% we are far from Table 2 on perplexity and on PIQA, HellaSwag, ARC-e and ARC-c. WinoGrande and OBQA are close.
- **MMLU drops faster in our runs.** At 30–50% we are near chance (25%).

### 5.3 The paper's own appendix agrees with us
The appendix ablations (Tables 5, 7, 8 and Fig. 6) appear to use the 20k-sample C4 setting; Table 7's averages equal Table 5's. Their numbers match ours, not Table 2:

| 50% pruning | WikiText-2 | Avg (7) |
|---|---|---|
| Paper Table 7 (joint pruning + fine-tuning) | 18.37 | 49.71 |
| Paper Table 8 (prune frequency 10, the default) | 18.21 | 49.59 |
| Paper Table 5 (20k C4) | – | 49.71 |
| **Ours, 40k** | **18.69** | **48.75** |
| **Ours, 40k + fine-tune** | **17.31** | **49.31** |
| Paper Table 2 (LaMini) | 11.60 | – |

At 20%, Table 5 gives an average of 60.05; ours is 60.6–61.1.

### 5.4 Why might Table 2 be better at high pruning ratios?
Neither the paper nor the code fully documents this. In order of likelihood:
1. **Much more training data.** As written, the released script uses all 2.58M LaMini samples: about 40,000 steps over 2 epochs, against our 624. Everything we measured fits this explanation:
   - perplexity at 50% kept improving with more data (21.4 at 20k → 18.7 at 40k → 17.3 at 60k with fine-tuning) and had not levelled off;
   - Fig. 6(b) says heavily pruned models need more steps to recover;
   - only Table 2 is better; the small-data ablations match ours.

   Against this: Table 6 reports about 2 hours in total at 14 s per step, which is only about 500 steps. That table is an efficiency benchmark and may not describe the Table 2 run.
2. **Different code version.** The repo has changed since publication; for example, there was a Llama-3 port. The PTB mismatch alone shows the published evaluation code is not exactly what produced the paper's numbers.
3. **Different evaluation setup** (lm-eval version, prompts). This explains part of the zero-shot gap but not the WikiText-2 gap, because our WikiText-2 baseline matches theirs.
4. **Unreported details** such as the number of runs or which checkpoint was chosen.

A direct test of explanation 1 would be to prune at 50% with 100k, 200k and 400k samples and check whether WikiText-2 keeps falling towards 11.6.

---

## 6. Conclusions
1. **LoRAPrune works:** removing 20% of LLaMA-7B costs about 2 perplexity points and about 5 zero-shot points.
2. **The 20% results reproduce** within 0.15–0.4 perplexity of the paper. **The 30–50% results reproduce the paper's appendix ablations, not its Table 2.**
3. **More training data clearly improves perplexity** (13–20% lower from 20k to 40k samples at 30–50%) **but barely moves zero-shot accuracy** (under 1 point except at 50%).
4. **fp16 vs bf16 and linear vs cosine schedules make little difference.** However, **fp16 is unsafe** with this training loop: a single overflow corrupts the model.
5. **Recovery fine-tuning after iterative pruning** (an experiment the paper does not run) helps slightly and consistently. At 50% it lowers WikiText-2 perplexity by 1.4 and raises accuracy by about 0.5 points. All of the gain comes within 50–100 steps; longer training starts to overfit. This differs from the paper's Table 7, where *one-shot* pruning followed by fine-tuning was worse than joint pruning.
6. **MMLU degrades fastest:** pruned models fall to 25–31%, close to chance. LaMini instruction data does not restore this knowledge.
7. **PTB numbers cannot be compared with the paper's** using the released code.

---

## 7. How to reproduce

```bash
conda activate loraprune
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # optional, once model + data are cached

# one pruning run (ratio 0.5, 40k samples, cosine) on GPU 0, then full evaluation
NSAMPLES=40000 LR_SCHED=cosine script/prune_ratio.sh 0.5 outputs_r50 0 > logs/prune_r50.log 2>&1
script/eval_all.sh r50 outputs_r50 0          # -> results/r50_{ppl.txt,zeroshot.json,mmlu.json}
script/eval_all.sh base "" 0                  # unpruned baseline
python script/summarize.py                    # -> results/SUMMARY.md

# whole experiments (3 ratios in parallel on GPUs 0-2)
script/run_40k.sh          # or script/run_20k_bf16.sh
script/run_finetune.sh     # needs run_40k/outputs_r{20,30,50}/adapter_model.bin
```

After each experiment, its `outputs_*`, `results/r*` and `logs/` files were moved into the matching `run_*/` folder.

Approximate time on one A100 80 GB:
- pruning: about 1 h per 312 steps (20k samples)
- evaluation: about 45 min per model (perplexity 2 min, zero-shot 10 min, MMLU 20–30 min)

---

## 8. Files in this branch
```
REPRODUCTION.md            this document
eval_lmeval.py             zero-shot / MMLU evaluation with lm-eval-harness
prune.py, inference.py     fixed (see §2)
loraprune/trainer.py       non-finite gradient guard
loraprune/utils.py         num_key_value_heads fix
script/                    run and evaluation scripts, summarize.py
results/                   unpruned baseline results (shared)
run_20k/                   20k fp16 linear      results/ logs/
run_20k_bf16/              20k bf16 cosine      results/ logs/
run_40k/                   40k bf16 cosine      results/ logs/
run_40k_fp16_failed/       NaN run (evidence for the fp16 bug)
run_finetune/              recovery fine-tuning results/ logs/ SUMMARY.md
```
