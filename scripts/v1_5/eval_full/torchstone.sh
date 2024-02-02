#!/bin/bash

SPLIT="touchstone_20230831"
MODEL_PATH=$1
python -m llava_hr.eval.model_vqa_torchstone \
    --model-path $1 \
    --question-file ./playground/data/eval/torchstone/$SPLIT.tsv \
    --answers-file ./playground/data/eval/torchstone/$SPLIT/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/torchstone/answers_upload/$SPLIT

python scripts/convert_torchstone_for_submission.py \
    --annotation-file ./playground/data/eval/torchstone/$SPLIT.tsv \
    --result-dir ./playground/data/eval/torchstone/$SPLIT \
    --upload-dir ./playground/data/eval/torchstone/$SPLIT \
    --experiment llava-v1.5-7b

python  ./playground/data/eval/torchstone/eval.py  ./playground/data/eval/torchstone/touchstone_20230831/llava-v1.5-7b.csv --model-name lavin_hr