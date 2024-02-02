#!/bin/bash
MODEL_PATH=$1
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
SPLIT="llava_caption_mscoco_test"
CKPT="llava-v1.5-7b"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava_hr.eval.model_caption_loader \
        --model-path $MODEL_PATH \
        --question-file ./playground/data/eval/coco-cap/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/coco-cap/val2014 \
        --answers-file ./playground/data/eval/coco-cap/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/coco-cap/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/coco-cap/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava_hr.eval.eval_caption \
    --annotation-file ./playground/data/eval/coco-cap/captions_test5k.json \
    --result-file ./playground/data/eval/coco-cap/$SPLIT/$CKPT/merge.jsonl
