CUDA_VISIBLE_DEVICES=0 python hard_prune_and_save.py \
    --base_model "huggyllama/llama-7b" \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
    --lora_weights 'outputs_oneshot_samples_20000_ratio_0.1' \
    --output_dir 'outputs_oneshot_samples_20000_ratio_0.1_pruned'
    