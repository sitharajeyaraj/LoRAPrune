import json
import os

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from loraprune.lora import Linear as LoraPruneLinear


def _replace_with_lora_linear(parent, child_name, out_features, in_features, r, lora_alpha, lora_dropout):
    new_module = LoraPruneLinear(
        in_features,
        out_features,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=False,
        merge_weights=False,
    )
    setattr(parent, child_name, new_module)


def load_pruned_model(
    pruned_dir: str,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    torch_dtype=torch.float16,
    device: str = "cuda",
):
    shape_map_path = os.path.join(pruned_dir, "layer_shape_map.json")
    if not os.path.exists(shape_map_path):
        raise FileNotFoundError(f"layer_shape_map.json not found in {pruned_dir}")

    with open(shape_map_path) as f:
        layer_shape_map = {int(k): v for k, v in json.load(f).items()}

    print("Building full-size model skeleton from config...")
    config = AutoConfig.from_pretrained(pruned_dir)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch_dtype)

    print("Reshaping layers to pruned dimensions...")
    for layer_idx, layer in enumerate(model.model.layers):
        shapes = layer_shape_map[layer_idx]
        r = shapes["lora_r"]

        q_out, q_in = shapes["q_proj_shape"]
        k_out, k_in = shapes["k_proj_shape"]
        v_out, v_in = shapes["v_proj_shape"]
        o_out, o_in = shapes["o_proj_shape"]
        g_out, g_in = shapes["gate_proj_shape"]
        u_out, u_in = shapes["up_proj_shape"]
        d_out, d_in = shapes["down_proj_shape"]

        _replace_with_lora_linear(layer.self_attn, "q_proj", q_out, q_in, r, lora_alpha, lora_dropout)
        _replace_with_lora_linear(layer.self_attn, "k_proj", k_out, k_in, r, lora_alpha, lora_dropout)
        _replace_with_lora_linear(layer.self_attn, "v_proj", v_out, v_in, r, lora_alpha, lora_dropout)
        _replace_with_lora_linear(layer.self_attn, "o_proj", o_out, o_in, r, lora_alpha, lora_dropout)
        _replace_with_lora_linear(layer.mlp, "gate_proj", g_out, g_in, r, lora_alpha, lora_dropout)
        _replace_with_lora_linear(layer.mlp, "up_proj", u_out, u_in, r, lora_alpha, lora_dropout)
        _replace_with_lora_linear(layer.mlp, "down_proj", d_out, d_in, r, lora_alpha, lora_dropout)

        # Bookkeeping fields the forward pass relies on for reshaping —
        # same fields prune_one_layer sets after physical pruning.
        head_dim = layer.self_attn.head_dim
        layer.self_attn.num_heads = shapes["num_heads"]
        layer.self_attn.hidden_size = q_out
        layer.self_attn.num_key_value_heads = k_out // head_dim
        layer.self_attn.num_key_value_groups = layer.self_attn.num_heads // layer.self_attn.num_key_value_heads
        layer.mlp.intermediate_size = g_out

    print("Loading pruned weights from safetensors...")
    weights_path = os.path.join(pruned_dir, "model.safetensors")
    state_dict = load_file(weights_path, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}...")

    model = model.to(torch_dtype)
    if device != "cpu":
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(pruned_dir)

    l0 = model.model.layers[0].self_attn.q_proj.weight.shape
    l10 = model.model.layers[10].self_attn.q_proj.weight.shape
    print(f"\nSanity check:")
    print(f"  Layer 0  q_proj (frozen): {l0}")
    print(f"  Layer 10 q_proj (pruned): {l10}")
    if l0 != l10:
        print(f"  Shapes differ — pruned architecture loaded correctly")
    else:
        print(f"  WARNING: shapes identical — pruning may not have loaded correctly")

    print(f"\nPruned model loaded successfully from {pruned_dir}")
    return model, tokenizer


if __name__ == "__main__":
    import fire
    fire.Fire(load_pruned_model)