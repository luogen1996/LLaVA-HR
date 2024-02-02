import os
import argparse
import json
import re

from llava_hr.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava_hr.eval.vqa import VQA
from llava_hr.eval.vqa_eval import VQAEval
import itertools
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--question-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file,question_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    results = [json.loads(line) for line in open(result_file)]
    pred_list = []
    for result in results: 
        pred_list.append({
            "answer": result['text'],
            "question_id": result['question_id'],
        })
    json.dump(pred_list, open(result_file.replace('jsonl','json'), 'w'), ensure_ascii=False)
    vqa = VQA(annotation_file,
              question_file)
    results = vqa.loadRes(
        resFile=result_file.replace('jsonl','json'),
        quesFile=question_file)
    vqa_scorer = VQAEval(vqa, results, n=2)
    vqa_scorer.evaluate()
    print(vqa_scorer.accuracy)

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file,args.question_file, args.result_file)

