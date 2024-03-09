#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
MODEL_PATH=$1

CKPT="llava-v1.5-7b"
SPLIT="test_q"
DATA_BASE="TGIF_Zero_Shot_QA"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava_hr.eval.model_videoqa_loader \
        --model-path $MODEL_PATH \
        --question-file ./playground/data/eval/$DATA_BASE/$SPLIT.json \
        --image-folder ./playground/data/eval/$DATA_BASE/images \
        --answers-file ./playground/data/eval/$DATA_BASE/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/$DATA_BASE/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/$DATA_BASE/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

rm -r ./playground/data/eval/$DATA_BASE/gpt

python -m llava_hr.eval.eval_video_qa \
    --pred_path $output_file \
    --output_dir ./playground/data/eval/$DATA_BASE/gpt \
    --output_json ./playground/data/eval/$DATA_BASE/results.json \
    --num_tasks 8

