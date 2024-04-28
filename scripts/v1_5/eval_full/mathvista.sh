#!/bin/bash
MODEL_PATH=$1
rm ./playground/data/eval/mathvista/results/bard/output_bard.json
CUDA_VISIBLE_DEVICES=0 python -m llava_hr.eval.model_mathvista \
    --model-path $MODEL_PATH \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava_hr.eval.extract_answer \
--output_dir ./playground/data/eval/mathvista/results/bard \
--output_file output_bard.json

rm ./playground/data/eval/mathvista/results/bard/scores_bard.json
python -m llava_hr.eval.calculate_score \
--output_dir ./playground/data/eval/mathvista/results/bard \
--output_file output_bard.json \
--score_file scores_bard.json
