import json
import os
import sys
from typing import List

import fire
import torch
from safetensors.torch import save_file

from loraprune.peft_model import get_peft_model
from loraprune.utils import freeze, prune_from_checkpoint
from loraprune.lora import LoraConfig

from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


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
    lora_weights: str = "",
    output_dir: str = "",
):
    assert base_model, "Please specify --base_model"
    assert lora_weights, "Please specify --lora_weights (directory with adapter_model.safetensors + lora_masks.pt)"
    assert output_dir, "Please specify --output_dir"

    tokenizer = AutoTokenizer.from_pretrained(base_model, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": 0},
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

    # ── Load adapter weights (same pattern as inference.py) ──
    safetensors_path = os.path.join(lora_weights, "adapter_model.safetensors")
    bin_path = os.path.join(lora_weights, "adapter_model.bin")
    pytorch_path = os.path.join(lora_weights, "pytorch_model.bin")

    from peft.peft_model import set_peft_model_state_dict

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
        raise FileNotFoundError(f"No adapter weights found in {lora_weights}")

    # ── Load and reinject masks (same pattern as inference.py) ──
    masks_path = os.path.join(lora_weights, "lora_masks.pt")
    if not os.path.exists(masks_path):
        print(f"WARNING: No lora_masks.pt found in {lora_weights} — aborting")
        sys.exit(1)

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

    model = model.to(device)

    # ── Physically prune (base weight, lora_A, lora_B all shrunk congruently) ──
    freeze(model)
    prune_from_checkpoint(model)

    # ── Build layer_shape_map.json ──
    layer_shape_map = {}
    for layer_idx, layer in enumerate(model.base_model.model.model.layers):
        q = layer.self_attn.q_proj
        k = layer.self_attn.k_proj
        v = layer.self_attn.v_proj
        o = layer.self_attn.o_proj
        g = layer.mlp.gate_proj
        u = layer.mlp.up_proj
        d = layer.mlp.down_proj

        head_dim = 128  # huggyllama/llama-7b: 4096 hidden / 32 heads
        num_heads = q.out_features // head_dim

        layer_shape_map[layer_idx] = {
            "q_proj_shape": [q.out_features, q.in_features],
            "k_proj_shape": [k.out_features, k.in_features],
            "v_proj_shape": [v.out_features, v.in_features],
            "o_proj_shape": [o.out_features, o.in_features],
            "gate_proj_shape": [g.out_features, g.in_features],
            "up_proj_shape": [u.out_features, u.in_features],
            "down_proj_shape": [d.out_features, d.in_features],
            "num_heads": num_heads,
            "lora_r": lora_r,
        }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "layer_shape_map.json"), "w") as f:
        json.dump(layer_shape_map, f, indent=2)
    print(f"Saved layer_shape_map.json ({len(layer_shape_map)} layers)")

    # ── Save raw pruned state dict (weight, lora_A, lora_B — no merge) ──
    # Strip the "base_model.model." prefix added by LoraPeftModelForCausalLM/
    # LoraModel wrapping, so keys line up with a plain LlamaForCausalLM layout
    # plus lora_A/lora_B suffixes for our custom loader to pick up.
    #
    # lora_mask keys are excluded: q/k/v/gate/up masks are already deleted by
    # prune_one_layer once pruning is applied, but o_proj/down_proj masks are
    # never deleted (they're stale all-ones leftovers, per inference.py's
    # documented behavior) — explicitly drop any lora_mask key so none of
    # that stale state leaks into the saved checkpoint.
    prefix = "base_model.model."
    raw_state_dict = model.state_dict()
    clean_state_dict = {}
    for key, tensor in raw_state_dict.items():
        if "lora_mask" in key:
            continue
        new_key = key[len(prefix):] if key.startswith(prefix) else key
        clean_state_dict[new_key] = tensor.data.cpu()

    save_file(clean_state_dict, os.path.join(output_dir, "model.safetensors"))
    print(f"Saved model.safetensors ({len(clean_state_dict)} tensors)")

    # ── Save base config + tokenizer for downstream loading ──
    model.base_model.model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nHard-pruned checkpoint saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)