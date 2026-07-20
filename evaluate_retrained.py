import os
import time
from typing import List

import fire
import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_file

from load_pruned_model import load_pruned_model

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


def main(
    pruned_dir: str = "",
    adapter_dir: str = "",
    cutoff_len: int = 128,
    ctx_length: int = 2048,
    batch_size: int = 8,
    cache_dir: str = None,
    eval_ppl_loraprune: bool = True,
    eval_ppl_irft: bool = True,
    eval_zero_shot: bool = True,
    eval_mmlu_5shot: bool = True,
    log_file: str = None,
):
    assert pruned_dir, "Please specify --pruned_dir (output of hard_prune_and_save.py)"
    assert adapter_dir, "Please specify --adapter_dir (output of retrain_pruned.py)"

    model, tokenizer = load_pruned_model(pruned_dir, torch_dtype=torch.float16, device=device)

    # ── Load retrained lora_A/lora_B weights onto the pruned skeleton ──
    adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    print(f"Loading retrained adapter from {adapter_path}")
    adapter_weights = load_file(adapter_path, device="cpu")
    loaded = 0
    for name, param in model.named_parameters():
        if name in adapter_weights:
            param.data = adapter_weights[name].to(param.device, dtype=param.dtype)
            loaded += 1
    print(f"Loaded {loaded} retrained LoRA tensors out of {len(adapter_weights)} saved")

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.half()
    model.eval()

    results_summary = {}
    log_output = []

    if log_file is None:
        log_file = os.path.join(adapter_dir, "evaluation_results.log")

    log_output.append(f"Pruned dir: {pruned_dir}")
    log_output.append(f"Adapter dir: {adapter_dir}")
    log_output.append("=" * 80)

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