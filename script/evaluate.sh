CUDA_VISIBLE_DEVICES=0 python inference.py \
    --base_model "huggyllama/llama-7b" \
    --lora_weights 'outputs_oneshot_samples_20000_ratio_0.1' \
    --cutoff_len 2048 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"