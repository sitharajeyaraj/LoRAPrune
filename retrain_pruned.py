import os
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

from load_pruned_model import load_pruned_model
from loraprune.lora import mark_only_lora_as_trainable

IGNORE_INDEX = -100


def train(
    pruned_dir: str = "",
    data_path: str = "",
    output_dir: str = "output_dir_retrained",
    nsamples: int = 20000,
    batch_size: int = 128,
    micro_batch_size: int = 2,
    num_epochs: int = 2,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    train_on_inputs: bool = True,
    group_by_length: bool = True,
    resume_from_checkpoint: str = None,
):
    assert pruned_dir, "Please specify --pruned_dir (output of hard_prune_and_save.py)"
    assert data_path, "Please specify --data_path"

    print(
        f"Retraining pruned model with params:\n"
        f"pruned_dir: {pruned_dir}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"nsamples: {nsamples}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
    )

    gradient_accumulation_steps = batch_size // micro_batch_size

    model, tokenizer = load_pruned_model(pruned_dir, torch_dtype=torch.bfloat16, device="cuda")

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "response": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                IGNORE_INDEX
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    # ── Only lora_A/lora_B are trainable; base weight stays frozen ──
    mark_only_lora_as_trainable(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable_params:,} || all params: {total_params:,} "
          f"|| trainable%: {100 * trainable_params / total_params:.4f}")

    if data_path.endswith(".json"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # ----------------------------------------------------------
    # FIXED (matches IRFT / main / prune.py convention): shuffle
    # the full dataset first with a fixed seed, THEN treat nsamples
    # as the TOTAL pool size (train+val combined), not train-only.
    # e.g. nsamples=20000, val_set_size=1000 -> 19000 train + 1000 val.
    # ----------------------------------------------------------
    data["train"] = data["train"].shuffle(seed=3407)
    if len(data["train"]) > nsamples:
        data["train"] = data["train"].select(range(nsamples))
        print(f"Limited dataset to {nsamples} samples "
              f"({nsamples - val_set_size} train + {val_set_size} val)")
    else:
        print(f"Dataset has {len(data['train'])} samples "
              f"(less than requested {nsamples}), using all.")

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=3407)
        train_data = train_val["train"].map(generate_and_tokenize_prompt)
        val_data = train_val["test"].map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=100 if val_set_size > 0 else None,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ── Save only the trainable lora_A/lora_B weights ──
    os.makedirs(output_dir, exist_ok=True)
    lora_weights = {k: v.data.cpu() for k, v in model.named_parameters() if "lora_" in k}
    from safetensors.torch import save_file
    save_file(lora_weights, os.path.join(output_dir, "adapter_model.safetensors"))
    print(f"Saved {len(lora_weights)} LoRA tensors to {output_dir}")

    print("\nRetraining complete.")


def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["response"]}"""


if __name__ == "__main__":
    fire.Fire(train)