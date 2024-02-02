#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

MODEL_VERSION=vicuna-v0-13b
# MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=v0
########### DO NOT CHANGE ###########

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data/vicuna/13B \
    --version $PROMPT_VERSION \
    --data_path ../data/cc3m_generated_sentences_v3.json \
    --image_folder ../data/images/gcc \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
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
    --is_robust True


deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data/vicuna/13B \
    --version $PROMPT_VERSION \
    --data_path ../data/llava_instruct_150k.json \
    --image_folder ../data/images/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_VERSION-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
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
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

python model_vqa.py \
    --model-path ./checkpoints/llava-$MODEL_VERSION-finetune \
    --question-file playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    ../data/images/val2014 \
    --answers-file \
    ./answer-file-our.jsonl

OPENAI_API_KEY="sk-1H6ZaZQVVsup5CQLXOeAIW4x3kz9E9ILTnNEfCoCC98AHh3u" python llava/eval/eval_gpt_review_visual.py \
    --question playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --context llava/eval/table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    playground/data/coco2014_val_qa_eval/qa90_gpt4_answer.jsonl \
    ./answer-file-our.jsonl \
    --rule llava/eval/table/rule.json \
    --output ./review.json

python llava/eval/summarize_gpt_review.py -f ./review.json