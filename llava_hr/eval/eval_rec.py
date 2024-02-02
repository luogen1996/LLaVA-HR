import os
import argparse
import json
import re

import torch
from torchvision.ops import box_iou



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    preds=[]
    gts=[]
    for line in open(args.result_file):
        line_=json.loads(line)
        preds.append(line_['pred'])
        gts.append(line_['ans'])
    target_boxes = torch.tensor(gts)
    pred_boxes = torch.tensor(preds)
    # normalized box value is too small, so that the area is 0.
    ious = box_iou(pred_boxes * 1000, target_boxes * 1000)
    ious = torch.einsum('i i -> i', ious)  # take diag elem
    correct = (ious > 0.5).sum().item()
    print('IoU@0.5: ', correct/len(gts)*100)
