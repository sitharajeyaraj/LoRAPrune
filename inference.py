import json
import sys
import os
import fire
import torch
import time
import transformers
import numpy as np
from typing import List
from peft.peft_model import set_peft_model_state_dict
from loraprune.peft_model import get_peft_model
from loraprune.utils import freeze, prune_from_checkpoint
from loraprune.lora import LoraConfig
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import (
    ZERO_SHOT_TASKS,
    evaluate_mmlu_5shot,
    evaluate_ppl,
    evaluate_zero_shot,
    extract_accuracy,
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    base_model: str = "",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.,
    lora_target_modules: List[str] = [
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj"
        ],
    lora_weights: str = "tloen/alpaca-lora-7b",
    cutoff_len: int = 128,
    ctx_length: int = 2048,
    batch_size: int = 8,
    cache_dir: str = None,
    eval_ppl_loraprune: bool = True,
    eval_ppl_irft: bool = True,
    eval_zero_shot: bool = True,
    eval_mmlu_5shot: bool = True,
    num_heads_per_layer: int = 32,
    log_file: str = None,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    tokenizer = AutoTokenizer.from_pretrained(base_model, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    if lora_weights:
        safetensors_path = os.path.join(lora_weights, "adapter_model.safetensors")
        bin_path         = os.path.join(lora_weights, "adapter_model.bin")
        pytorch_path     = os.path.join(lora_weights, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            print(f"Loading LoRA weights from {safetensors_path}")
            from safetensors.torch import load_file
            adapters_weights = load_file(safetensors_path)
            set_peft_model_state_dict(model, adapters_weights)
        elif os.path.exists(bin_path):
            print(f"Loading LoRA weights from {bin_path}")
            adapters_weights = torch.load(bin_path, map_location="cpu")
            set_peft_model_state_dict(model, adapters_weights)
        elif os.path.exists(pytorch_path):
            print(f"Loading LoRA weights from {pytorch_path}")
            adapters_weights = torch.load(pytorch_path, map_location="cpu")
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"No LoRA weights found in {lora_weights} — running with unmodified LoRA init")

        masks_path = os.path.join(lora_weights, "lora_masks.pt")
        if os.path.exists(masks_path):
            print(f"Loading masks from {masks_path}")
            masks = torch.load(masks_path, map_location="cpu", weights_only=False)
            reinjected = 0
            for name, module in model.named_modules():
                if name in masks:
                    module.lora_mask = torch.nn.Parameter(
                        masks[name].to(next(module.parameters()).device),
                        requires_grad=False
                    )
                    reinjected += 1
            print(f"Reinjected {reinjected} masks out of {len(masks)} saved")
        else:
            print(f"WARNING: No lora_masks.pt found in {lora_weights}")
            print(f"prune_from_checkpoint will fail without masks — aborting")
            sys.exit(1)

    # ──────────────────────────────────────────────────────────
    # Set up logging BEFORE freeze()/prune_from_checkpoint(), since
    # prune_one_layer deletes q_proj/k_proj/v_proj/gate_proj/up_proj
    # lora_mask attributes once pruning is applied. This is the last
    # point where the real pruning decisions still exist as tensors.
    # ──────────────────────────────────────────────────────────
    results_summary = {}
    log_output = []

    if log_file is None:
        log_file = os.path.join(lora_weights, "evaluation_results.log") if os.path.isdir(lora_weights) else "evaluation_results.log"

    log_output.append(f"Base model: {base_model}")
    log_output.append(f"LoRA weights: {lora_weights}")
    log_output.append("=" * 80)

    # ──────────────────────────────────────────────────────────
    # Mask logging — dump the pruning mask for every module that
    # had one reinjected, with layer name + raw 0/1 values.
    #
    # Confirmed against loraprune/utils.py (local_prune):
    #   - q_proj/k_proj/v_proj masks ARE head-grouped: length =
    #     out_features (e.g. 4096), but values are constant within
    #     each head_dim-sized block (128 for llama-7b), since
    #     local_prune decides per-head then broadcasts. Safe to
    #     reduce these to num_heads_per_layer values.
    #   - gate_proj/up_proj masks are PLAIN PER-CHANNEL over
    #     intermediate_size (e.g. 11008) — NOT head-grouped, so we
    #     do NOT attempt a per-head reduction for these even though
    #     the length may happen to be divisible by num_heads_per_layer.
    #   - o_proj/down_proj are excluded: their lora_mask is leftover
    #     all-ones initialization from loraprune/lora.py's Linear.__init__,
    #     never updated by local_prune (they're pruned via q_proj's/
    #     gate_proj's masks instead). Logging them would show stale
    #     noise, not a real pruning decision.
    # ──────────────────────────────────────────────────────────
    EXCLUDED_SUFFIXES = ("o_proj", "down_proj")
    HEAD_GROUPED_SUFFIXES = ("q_proj", "k_proj", "v_proj")

    log_output.append("\n=== Pruning Masks (captured pre-pruning, real decisions) ===")
    print("\n=== Pruning Masks (captured pre-pruning, real decisions) ===")
    for name, module in model.named_modules():
        if name.split(".")[-1] in EXCLUDED_SUFFIXES:
            continue
        mask_tensor = getattr(module, "lora_mask", None)
        if mask_tensor is None:
            continue
        flat = mask_tensor.detach().cpu().reshape(-1)
        raw_values = [int(v) for v in (flat != 0).long().tolist()]

        n_pruned = raw_values.count(0)
        line = f"{name}  (len={len(raw_values)}, pruned={n_pruned})  raw_mask={raw_values}"
        print(line)
        log_output.append(line)

        is_head_grouped = name.split(".")[-1] in HEAD_GROUPED_SUFFIXES
        if is_head_grouped and len(raw_values) % num_heads_per_layer == 0:
            per_head_size = len(raw_values) // num_heads_per_layer
            head_mask = []
            for h in range(num_heads_per_layer):
                chunk = raw_values[h * per_head_size:(h + 1) * per_head_size]
                head_mask.append(1 if any(chunk) else 0)
            n_heads_pruned = head_mask.count(0)
            head_line = f"{name}  per_head({num_heads_per_layer} heads, pruned={n_heads_pruned})={head_mask}"
            print(head_line)
            log_output.append(head_line)

    model = model.to(device)

    freeze(model)
    prune_from_checkpoint(model)

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.half()

    model.eval()

    # ──────────────────────────────────────────────────────────
    # PPL method 1: LoRAPrune-native (chunked by cutoff_len,
    # non-overlapping windows, mean NLL over all chunks)
    # ──────────────────────────────────────────────────────────
    if eval_ppl_loraprune:
        from torch.utils.data.dataset import Dataset

        times = []

        class IndexDataset(Dataset):
            def __init__(self, tensors):
                self.tensors = tensors

            def __getitem__(self, index):
                return self.tensors[index]

            def __len__(self):
                return len(self.tensors)

        def process_data(samples, tokenizer, seq_len, field_name):
            test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
            test_ids_batch = []
            nsamples = test_ids.numel() // seq_len

            for i in range(nsamples):
                batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
                test_ids_batch.append(batch)
            test_ids_batch = torch.stack(test_ids_batch)
            return IndexDataset(tensors=test_ids_batch)

        @torch.no_grad()
        def llama_eval(model, loader, device):
            model.eval()
            nlls = []
            for batch in loader:
                batch = batch.to(device)
                with torch.cuda.amp.autocast():
                    t1 = time.time()
                    output = model(batch)
                    times.append(time.time() - t1)
                lm_logits = output.logits

                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
            ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
            return ppl.item()

        print("\n=== LoRAPrune-native PPL (cutoff_len={}) ===".format(cutoff_len))

        eval_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(eval_data, tokenizer, cutoff_len, 'text')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        wt2_ppl = llama_eval(model, test_loader, device)
        mean_time = np.mean(times) if times else 0.0
        msg = "wikitext2 ppl (loraprune method): {:.2f}  inference time: {:2f}".format(wt2_ppl, mean_time)
        print(msg)
        log_output.append(msg)
        results_summary["ppl_wikitext2_loraprune"] = round(wt2_ppl, 2)

        times.clear()
        eval_data = load_dataset('ptb_text_only', 'penn_treebank', split='test', trust_remote_code=True)
        test_dataset = process_data(eval_data, tokenizer, cutoff_len, 'sentence')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        ptb_ppl = llama_eval(model, test_loader, device)
        mean_time = np.mean(times) if times else 0.0
        msg = "PTB ppl (loraprune method): {:.2f}  inference time: {:2f}".format(ptb_ppl, mean_time)
        print(msg)
        log_output.append(msg)
        results_summary["ppl_ptb_loraprune"] = round(ptb_ppl, 2)

    # ──────────────────────────────────────────────────────────
    # PPL method 2: IRFT's evaluate_ppl (same method used to
    # score IRFT checkpoints — for fair, apples-to-apples
    # comparison, using ctx_length instead of cutoff_len)
    # ──────────────────────────────────────────────────────────
    if eval_ppl_irft:
        msg = "\n=== IRFT-method PPL (ctx_length={}) ===".format(ctx_length)
        print(msg)
        log_output.append(msg)
        for dataset in ["wikitext"]:
            ppl = evaluate_ppl(model, tokenizer, dataset, ctx_length, device, cache_dir)
            results_summary[f"ppl_{dataset}_irft"] = round(ppl, 2)
            msg = f"PPL {dataset} (irft method): {ppl:.2f}"
            print(msg)
            log_output.append(msg)

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

    if eval_mmlu_5shot:
        msg = "\nRunning MMLU 5-shot evaluation..."
        print(msg)
        log_output.append(msg)
        mmlu_results = evaluate_mmlu_5shot(model, tokenizer, device, batch_size, cache_dir)
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

    msg = "\n=== Summary ==="
    print(msg)
    log_output.append(msg)
    for key, value in results_summary.items():
        msg = f"  {key:<25} {value}"
        print(msg)
        log_output.append(msg)

    keys = list(results_summary.keys())
    values = list(results_summary.values())
    msg = "\n=== LaTeX row ==="
    print(msg)
    log_output.append(msg)
    msg = " & ".join(str(key) for key in keys) + " \\\\"
    print(msg)
    log_output.append(msg)
    msg = " & ".join(str(value) for value in values) + " \\\\"
    print(msg)
    log_output.append(msg)

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "w") as f:
        f.write("\n".join(log_output))
    print(f"\n✓ Results saved to: {log_file}")

    return results_summary


if __name__ == "__main__":
    fire.Fire(main)