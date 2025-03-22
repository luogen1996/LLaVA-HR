#!/bin/bash

export TRANSFORMERS_OFFLINE=1 # 设置为1时，启用Transformers的离线模式
### deepspeed单机多卡设置显卡，不要使用export CUDA_VISIBLE_DEVICES=2,5，改成deepspeed --include localhost:2,5
include=localhost:0,1,2,3 # 设置显卡id

model_name_or_path=/mnt/share/xujing/checkpoints/ChartMLLM # 模型名称
# data_path=/mnt/share/xujing/flux/llava_format_description.json # 训练的json
image_folder=/mnt/share/xujing/flux/version3 # 训练的图像数据
data_path=/mnt/share/xujing/flux/train_dataset_3.1.json
# image_folder=/mnt/share/xujing/chartqa/ChartQA_Dataset/train/png
    # --lora_enable True --lora_r 8 --lora_alpha 16 \
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" deepspeed --master_port 29501 --include $include llava_hr/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True --lora_r 8 --lora_alpha 16 \
    --model_name_or_path $model_name_or_path  \
    --version v1 \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower_checkpoint /mnt/share/xujing/LLaVA-HR/checkpoints/multipath_visiontower.pth \
    --pretrain_mm_mlp_adapter /mnt/share/xujing/LLaVA-HR/checkpoints/multipath_visiontower.pth \
    --is_multipath_encoder True \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_slow convnext_xxlarge.clip_laion2b_soup \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/llava-mllm-dataset3.1-lorabackbone-bs64 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 3 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --freeze_vision False \
    --input_image_size 1024