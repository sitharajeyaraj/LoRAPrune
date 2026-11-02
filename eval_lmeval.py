"""Zero-shot / few-shot evaluation of a LoRAPrune model with lm-evaluation-harness.

The pruned model has different widths per layer, so it cannot be saved as a normal
HF checkpoint. Instead we rebuild it exactly like inference.py (base model + LoRA
weights + masks -> physically pruned, LoRA merged) and hand the model object to lm-eval.

Usage:
  python eval_lmeval.py --base_model huggyllama/llama-7b --lora_weights outputs_dir \
      --tasks boolq,piqa --num_fewshot 0 --output results/r50_zeroshot.json
  (omit --lora_weights / pass "" to evaluate the unpruned base model)
"""
import json
import os
from typing import List

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from loraprune.lora import LoraConfig
from loraprune.peft_model import get_peft_model
from loraprune.utils import freeze, prune_from_checkpoint


def load_pruned_model(base_model, lora_weights, lora_r, lora_alpha, lora_target_modules):
    tokenizer = AutoTokenizer.from_pretrained(base_model, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto"
    )
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    if lora_weights:
        checkpoint_name = os.path.join(lora_weights, "adapter_model.bin")
        adapters_weights = torch.load(checkpoint_name)
        for name, param in adapters_weights.items():
            if "lora_mask" in name:
                adapters_weights[name] = param.reshape(-1)
        result = model.load_state_dict(adapters_weights, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected keys in checkpoint: {result.unexpected_keys[:5]}")
        missing_lora = [k for k in result.missing_keys if "lora_" in k]
        if missing_lora:
            raise RuntimeError(f"LoRA weights missing from checkpoint: {missing_lora[:5]}")
        print(f"Loaded {len(adapters_weights)} LoRA tensors from {checkpoint_name}")
    model = model.to("cuda")
    freeze(model)
    prune_from_checkpoint(model)  # physically removes pruned heads/channels and merges LoRA
    model.eval()

    hf_model = model.base_model.model  # PeftModel -> LoraModel -> LlamaForCausalLM
    n_params = sum(p.numel() for p in hf_model.parameters())
    print(f"Model parameters after pruning: {n_params / 1e9:.3f}B")
    return hf_model, tokenizer, n_params


def main(
    base_model: str = "huggyllama/llama-7b",
    lora_weights: str = "",
    tasks: str = "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa",
    num_fewshot: int = 0,
    output: str = "results/lmeval.json",
    batch_size: str = "auto",
    limit: float = None,  # for quick tests only
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_target_modules: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    if isinstance(tasks, str):
        tasks = tasks.split(",")
    if isinstance(lora_target_modules, str):
        lora_target_modules = lora_target_modules.split(",")

    hf_model, tokenizer, n_params = load_pruned_model(
        base_model, lora_weights, lora_r, lora_alpha, lora_target_modules
    )
    lm = HFLM(pretrained=hf_model, tokenizer=tokenizer, batch_size=batch_size, max_length=2048)
    results = lm_eval.simple_evaluate(
        model=lm, tasks=list(tasks), num_fewshot=num_fewshot, limit=limit, log_samples=False
    )

    print(lm_eval.utils.make_table(results))
    if "groups" in results:
        print(lm_eval.utils.make_table(results, "groups"))

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(
            {
                "base_model": base_model,
                "lora_weights": lora_weights,
                "num_fewshot": num_fewshot,
                "n_params": n_params,
                "results": results["results"],
                "groups": results.get("groups", {}),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Saved results to {output}")


if __name__ == "__main__":
    fire.Fire(main)
