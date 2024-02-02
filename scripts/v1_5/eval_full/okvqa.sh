#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_PATH=$1
CKPT="llava-v1.5-7b"
SPLIT="okvqa_val"
OCRVQADIR="./playground/data/eval/okvqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava_hr.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file ./playground/data/eval/okvqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/coco-cap/val2014 \
        --answers-file ./playground/data/eval/okvqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/okvqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/okvqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava_hr.eval.eval_okvqa \
    --annotation-file ./playground/data/eval/okvqa/mscoco_val2014_annotations.json \
    --question-file ./playground/data/eval/okvqa/OpenEnded_mscoco_val2014_questions.json \
    --result-file $output_file