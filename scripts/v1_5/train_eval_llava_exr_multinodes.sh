#!/bin/bash


deepspeed llava_hr/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data/vicuna/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k_nopadding.json \
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
    --output_dir ./checkpoints/llava-exr-7b-sft-8k-px-fix \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
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
   --image_grid_pinpoints [[2048,4096],[4096,2048],[2048,3072],[3072,2048],[6144,1024],[1024,6144],[3072,3072],[8192,1024],[1024,8192]]

#  --image_grid_pinpoints [[1024,12288],[12288,1024],[4096,3072],[3072,4096],[2048,6144],[6144,2048]]

bash scripts/v1_5/eval.sh ./checkpoints/log-llava-hr-7b-sft-1024-gemini-augv9-802 2>&1 | tee log-llava-hr-7b-sft-1024-gemini-augv9-802.txt

#stage 3 script
deepspeed llava_hr/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/llava-hr-7b-pretrain-1024-stage2-3k-sigclip_convxxl \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k_nopadding.json \
    --image_folder ./playground/data/mini_gemini_data \
    --vision_tower google/siglip-so400m-patch14-384 \
    --vision_tower_slow convnext_xxlarge.clip_laion2b_soup \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-hr-7b-sft-1024-stage3-3k-sigclip_convxxl \
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
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --is_multipath_encoder True \
	--mm_patch_merge_type spatial_unpad \
	--freeze_vision False \
	--input_image_size 1024 \
    --image_grid_pinpoints [[1024,2048],[2048,1024],[2048,2048],[3072,1024],[1024,3072],[2048,4096],[4096,2048],[2048,3072],[3072,2048],[6144,1024],[1024,6144],[8192,1024],[1024,8192]]

#stage 2 script
export DEEPSPEED_PORT=1172
deepspeed --num_nodes 3 --num_gpus 8  --master_addr 10.24.116.20 --master_port=2347   --hostfile=./hostfile.txt \
    llava_hr/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data/vicuna/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k_nopadding.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-hr-7b-pretrain-384-lowres-v2/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/llava-exr-7b-pretrain-8k-final-mnodes \
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
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --is_multipath_encoder True \
	--freeze_vision False \
	--input_image_size 1024 \
    --image_grid_pinpoints [[1024,2048],[2048,1024],[2048,2048],[3072,1024],[1024,3072],[2048,4096],[4096,2048],[2048,3072],[3072,2048],[6144,1024],[1024,6144],[8192,1024],[1024,8192]]




deepspeed --num_nodes 2 --num_gpus 8  --master_addr 10.24.116.20 --master_port=9901   --hostfile=./hostfile.txt \
    llava_hr/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data/vicuna/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/mini_gemini_data/Mini-Gemini-Pretrain/minigemini_pretrain_filter_highres_ocr_pdf_tcap.json \
    --image_folder ./playground/data/mini_gemini_data \
    --vision_tower google/siglip-so400m-patch14-384 \
    --vision_tower_slow convnext_xxlarge.clip_laion2b_soup \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-hr-7b-pretrain-384-lowres-v3-siglip_convxxl/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/llava-exr-7b-pretrain-8k-final \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
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
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --is_multipath_encoder True \
	--freeze_vision False \
	--input_image_size 1024 \
    --image_grid_pinpoints [[1024,2048],[2048,1024],[2048,2048],[3072,1024],[1024,3072],[2048,4096],[4096,2048],[2048,3072],[3072,2048],[6144,1024],[1024,6144],[8192,1024],[1024,8192]]
