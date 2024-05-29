#!/bin/bash


deepspeed llava_hr/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data/vicuna/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k_left_pad.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-hr-7b-pretrain-384-px/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-exr-7b-sft-3k-px \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5194 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --is_multipath_encoder True \
	--mm_patch_merge_type spatial_unpad \
	--freeze_vision False \
	--input_image_size 1024 \
    --image_grid_pinpoints [[1024,2048],[2048,1024],[2048,2048],[3072,1024],[1024,3072]]

bash scripts/v1_5/eval.sh ./checkpoints/llava-exr-7b-sft-3k-px 2>&1 | tee log-llava-exr-7b-sft-3k-px.txt