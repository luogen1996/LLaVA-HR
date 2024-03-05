#!/bin/bash




deepspeed llava_hr/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data/vicuna/vicuna-13b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_slow convnext_xxlarge.clip_laion2b_soup \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-hr-x-13b-pretrain-384/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-hr-x-13b-sft-768 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2496 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --is_multipath_encoder True \
    --freeze_vision False \
    --input_image_size 1024

bash scripts/v1_5/eval.sh ./checkpoints/llava-hr-x-13b-sft-1024 2>&1 | tee log-llava-hr-x-13b-sft-1024.txt