#!/bin/bash



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval_full/ocrvqa.sh  ./checkpoints/llava-v1.5-7b-vitstconv-deep-final_s2f_1024/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval_full/ocrvqa.sh  ./checkpoints/llava-v1.5-13b-vitstconv-deep-final_s2f_1024