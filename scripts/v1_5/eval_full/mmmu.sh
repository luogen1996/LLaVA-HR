#!/bin/bash
MODEL_PATH=$1
python -m llava_hr.eval.model_mmmu \
    --model-path $MODEL_PATH \
    --output_path  ./playground/data/eval/MMMU/llava1.5_13b_val.json \
    --data_path ./playground/data/eval/MMMU \
    --conv-mode vicuna_v1 \
    --split validation

python -m llava_hr.eval.eval_mmmu_only \
    --output_path ./playground/data/eval/MMMU/llava1.5_13b_val.json \
    --answer_path ./playground/data/eval/MMMU/answer_dict_val.json

