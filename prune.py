import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from loraprune.trainer import LoRAPruneTrainer
from loraprune.utils import freeze
from loraprune.lora import LoraConfig

from peft import (
    prepare_model_for_kbit_training,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.peft_model import get_peft_model_state_dict, set_peft_model_state_dict

IGNORE_INDEX = -100


def train(
    # model/data params
    base_model: str = "",  # the required argument
    data_path: str = "",  # the required argument
    output_dir: str = "output_dir",
    # training hyperparams
    nsamples: int = 25000,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    lr_scheduler_type: str = "linear",  # paper uses cosine
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # pruning hyperparams
    ratio: float = 0.5,
    init_ratio: float = 0,
    warmup_iters: float = 0.1,
    cooldown_iters: float = 0.1,
    prune_freq: int = 10,
    prune_metric: str = 'lora',  # options: lora|grad|magnitude
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj"
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    load_in_8bit: bool = False,
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    # recovery fine-tuning after pruning
    init_lora: str = "",  # adapter_model.bin of a pruned model: load LoRA + masks, masks stay fixed
    data_offset: int = 0,  # skip this many shuffled samples (use data not seen during pruning)
    lora_save_steps: int = 0,  # also save LoRA + masks every N steps to output_dir/step-N
):
    print(
        f"Pruning with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"nsamples: {nsamples}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"lr_scheduler_type: {lr_scheduler_type}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"prune_metric: {prune_metric}\n"
        f"init_lora: {init_lora}\n"
        f"data_offset: {data_offset}\n"
        f"lora_save_steps: {lora_save_steps}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
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
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    # TODO: convert sparseLinear for model here
    # utils.convert_sparse_network(model, target_modules=lora_target_modules)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        peft_type="LORA"
    )
    from loraprune.peft_model import get_peft_model
    # from peft import get_peft_model
    model = get_peft_model(model, config)

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # randomly subsample nsamples training examples (plus the validation set); <= 0 uses all data
    if nsamples > 0:
        n = min(nsamples + val_set_size, len(data["train"]) - data_offset)
        data["train"] = data["train"].shuffle(seed=42).select(range(data_offset, data_offset + n))
        print(f"Using {n} samples ({n - val_set_size} train + {val_set_size} val), offset {data_offset}")

    freeze(model)
    if init_lora:
        # recovery fine-tuning: start from a pruned model and never change its masks
        sd = torch.load(init_lora, map_location="cpu")
        res = torch.nn.Module.load_state_dict(model, sd, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys[:5]
        missing = [k for k in res.missing_keys if "lora_" in k]
        assert not missing, missing[:5]
        masks = [v for k, v in sd.items() if "lora_mask" in k]
        print(f"Loaded {len(sd)} tensors from {init_lora}; "
              f"{sum(int((m == 0).sum()) for m in masks)} masked channels in {len(masks)} masks")
        init_ratio = ratio  # schedule never exceeds init_ratio -> local_prune is never called
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # utils.print_trainable_parameters(model)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    def save_lora(out):
        state = {k: v.detach().cpu().contiguous()
                 for k, v in torch.nn.Module.state_dict(model).items() if "lora_" in k}
        os.makedirs(out, exist_ok=True)
        torch.save(state, os.path.join(out, "adapter_model.bin"))
        return state

    class SaveLoraCallback(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if lora_save_steps and state.global_step % lora_save_steps == 0:
                save_lora(os.path.join(output_dir, f"step-{state.global_step}"))

    trainer = LoRAPruneTrainer(
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
            lr_scheduler_type=lr_scheduler_type,
            bf16=True,  # was fp16; the custom training loop has no GradScaler overflow handling
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=100 if val_set_size > 0 else None,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # data_collator=data_collator,
        ratio=ratio,
        init_ratio=init_ratio,
        warmup_iters=warmup_iters,
        cooldown_iters=cooldown_iters,
        prune_freq=prune_freq,
        prune_metric=prune_metric
    )
    trainer.add_callback(SaveLoraCallback())

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # save LoRA weights + pruning masks directly; peft>=0.13 save_pretrained does not
    # understand loraprune's config object and writes empty files instead
    lora_state = save_lora(output_dir)
    n_masks = sum("lora_mask" in k for k in lora_state)
    print(f"Saved {len(lora_state)} LoRA tensors ({n_masks} pruning masks) to {output_dir}/adapter_model.bin")

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["response"]}"""


if __name__ == "__main__":
    fire.Fire(train)