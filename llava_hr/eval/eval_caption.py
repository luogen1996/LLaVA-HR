import os
import argparse
import json
import re

from llava_hr.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import tempfile
class COCOEvaler(object):
    def __init__(self, annfile):
        super(COCOEvaler, self).__init__()
        self.coco = COCO(annfile)
        if not os.path.exists('./tmp'):
            os.mkdir('./tmp')

    def eval(self, result):
        in_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir='./tmp')
        json.dump(result, in_file)
        in_file.close()

        cocoRes = self.coco.loadRes(in_file.name)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate()
        os.remove(in_file.name)
        return cocoEval.eval
def jsonl2json(pred):
    data=[]
    for pred_ in pred:
        data.append(pred_)
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()

    evaler = COCOEvaler(args.annotation_file)
    preds= [json.loads(line) for line in open(args.result_file)]
    preds=jsonl2json(preds)
    json.dump(preds,open('./tmp/preds,json','w'))
    res=evaler.eval(json.load(open('./tmp/preds,json')))