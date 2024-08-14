#!/bin/bash
export BNB_CUDA_VERSION=117

cd /data/luogen_code/LLaVA-HR

deepspeed --num_nodes 13 --num_gpus 8  --master_addr 10.24.116.46 --master_port=2347 --hostfile=./hostfile.txt \
    llava_hr/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/82_store/luogen/vicuna-7b-v1.5/ \
    --version v1 \
    --data_path /mnt/82_store/luogen/minigemini_pretrain_filter_highres_ocr_pdf_tcap.json \
    --image_folder /data/luogen/ \
    --vision_tower /mnt/82_store/luogen/siglip-so400m-patch14-384 \
    --vision_tower_slow convnext_xxlarge.clip_laion2b_soup \
    --pretrain_mm_mlp_adapter /mnt/82_store/luogen/checkpoints/llava-exr-7b-pretrain-stage1/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir /mnt/82_store/luogen/checkpoints/llava-exr-7b-pretrain-stage2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 4e-5 \
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
	--input_image_size 768 \
    --image_grid_pinpoints [[1024,1024],[1024,2048],[2048,1024],[2048,2048],[3072,1024],[1024,3072],[2048,3072],[3072,2048],[6144,1024],[1024,6144]]