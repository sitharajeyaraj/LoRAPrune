import os
import fire
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# PPL — sliding window over a single concatenated document
# ---------------------------------------------------------------------------

def evaluate_ppl(model, tokenizer, dataset_name, ctx_length, device, cache_dir):
    if dataset_name == "wikitext":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir)
        text = "\n\n".join(data["text"])
    else:
        raise ValueError(f"Unknown PPL dataset: {dataset_name}")

    # truncation=False: encode the full corpus; the sliding window below feeds
    # only ctx_length tokens to the model at a time, so no indexing errors occur.
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    seq_len = encodings.input_ids.size(1)
    stride = ctx_length

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride), desc=f"PPL ({dataset_name})"):
        end_loc = min(begin_loc + ctx_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl


# ---------------------------------------------------------------------------
# Zero-shot accuracy via lm-evaluation-harness (v0.4+)
# ---------------------------------------------------------------------------

ZERO_SHOT_TASKS = [ 
    "boolq",
    "piqa",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
]


def evaluate_zero_shot(model, tokenizer, device, batch_size, cache_dir):
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError("lm_eval not found. Install with: pip install lm-eval>=0.4.0")

    model.config.use_cache = False  # avoids deprecated tuple past_key_values in lm_eval
    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device, batch_size=batch_size)

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=ZERO_SHOT_TASKS,
        num_fewshot=0,
        batch_size=batch_size,
        cache_requests=True if cache_dir else False,
    )
    return results


def evaluate_mmlu_5shot(model, tokenizer, device, batch_size, cache_dir):
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError("lm_eval not found. Install with: pip install lm-eval>=0.4.0")

    model.config.use_cache = False
    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device, batch_size=batch_size)

    return evaluator.simple_evaluate(
        model=lm,
        tasks=["mmlu"],
        num_fewshot=5,
        batch_size=batch_size,
        cache_requests=True if cache_dir else False,
    )


def extract_accuracy(results, key):
    for section in ("results", "groups"):
        values = results.get(section, {}).get(key, {})
        if not isinstance(values, dict):
            continue
        acc = values.get("acc_norm,none")
        if acc is None:
            acc = values.get("acc,none")
        if acc is not None:
            return round(acc * 100, 2)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    model_path: str = "outputs_fpft",
    device: str = "cuda:0",
    cache_dir: str = None,
    ctx_length: int = 512,
    batch_size: int = 8,
    eval_ppl: bool = True,
    eval_zero_shot: bool = True,
    eval_mmlu_5shot: bool = True,
    log_file: str = None,
):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.pad_token_id or 0

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        cache_dir=cache_dir,
    )
    model.config.use_cache = True
    model.eval()

    results_summary = {}
    
    # Prepare log file if not specified
    if log_file is None:
        log_file = os.path.join(os.path.dirname(model_path) or ".", "evaluation_results.log")
    
    log_output = []
    
    # Add model path to log
    log_output.append(f"Model Path: {model_path}")
    log_output.append("=" * 80)

    # --- PPL ---
    if eval_ppl:
        for ds in ["wikitext"]:
            ppl = evaluate_ppl(model, tokenizer, ds, ctx_length, device, cache_dir)
            results_summary[f"ppl_{ds}"] = round(ppl, 2)
            msg = f"PPL {ds}: {ppl:.2f}"
            print(msg)
            log_output.append(msg)

    # --- Zero-shot ---
    if eval_zero_shot:
        msg = "\nRunning zero-shot evaluations..."
        print(msg)
        log_output.append(msg)
        zs_results = evaluate_zero_shot(model, tokenizer, device, batch_size, cache_dir)

        msg = "\n=== Zero-shot Results ==="
        print(msg)
        log_output.append(msg)
        for task in ZERO_SHOT_TASKS:
            acc_pct = extract_accuracy(zs_results, task) or 0.0
            results_summary[task] = acc_pct
            msg = f"  {task:<20} {acc_pct:.2f}%"
            print(msg)
            log_output.append(msg)

    # --- MMLU 5-shot ---
    if eval_mmlu_5shot:
        msg = "\nRunning MMLU 5-shot evaluation..."
        print(msg)
        log_output.append(msg)
        mmlu_results = evaluate_mmlu_5shot(
            model, tokenizer, device, batch_size, cache_dir
        )
        acc_pct = extract_accuracy(mmlu_results, "mmlu")
        if acc_pct is None:
            subject_scores = []
            for task, values in mmlu_results.get("results", {}).items():
                if not task.startswith("mmlu_"):
                    continue
                score = values.get("acc,none")
                if score is not None:
                    subject_scores.append(score)
            acc_pct = round(100 * sum(subject_scores) / len(subject_scores), 2)
        results_summary["mmlu_5shot"] = acc_pct
        msg = f"  {'mmlu_5shot':<20} {acc_pct:.2f}%"
        print(msg)
        log_output.append(msg)

    # --- Summary ---
    msg = "\n=== Summary ==="
    print(msg)
    log_output.append(msg)
    for k, v in results_summary.items():
        msg = f"  {k:<25} {v}"
        print(msg)
        log_output.append(msg)
    
    keys = list(results_summary.keys())
    vals = list(results_summary.values())
    msg = "\n=== LaTeX row ==="
    print(msg)
    log_output.append(msg)
    msg = " & ".join(str(k) for k in keys) + " \\\\"
    print(msg)
    log_output.append(msg)
    msg = " & ".join(str(v) for v in vals) + " \\\\"
    print(msg)
    log_output.append(msg)
    
    # Save results to file
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "w") as f:
        f.write("\n".join(log_output))
    msg = f"\n✓ Results saved to: {log_file}"
    print(msg)
    
    return results_summary


if __name__ == "__main__":
    fire.Fire(main)