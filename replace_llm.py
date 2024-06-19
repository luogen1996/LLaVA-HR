import transformers
import json
from copy import deepcopy
import torch
from llava_hr.model import *
from llava_hr.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava_hr.model.builder import load_pretrained_model

model_path = './checkpoints/llava-hr-7b-pretrain-1024-stage2-v2'
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,None,model_name)

vicuna = transformers.AutoModelForCausalLM.from_pretrained('/data/vicuna/vicuna-7b-v1.5')


import torch.nn as nn
import torch.nn.init as init
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        # Initialize weights using Xavier/Glorot initialization
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0.0)
    if isinstance(module, nn.LayerNorm):
        init.constant_(module.weight, 1.0)
        init.constant_(module.bias, 0.0)

new_backbone = torch.nn.ModuleList()

for i in range(32):
    new_backbone.append(deepcopy(vicuna.model.layers[i]))

model.model.layers = new_backbone
vicuna_lm_head = deepcopy(vicuna.lm_head)
vicuna_embed_tokens = deepcopy(vicuna.model.embed_tokens)
vicuna_norm = deepcopy(vicuna.model.norm)
model.model.lm_head = vicuna_lm_head
model.model.embed_tokens = vicuna_embed_tokens
model.model.norm = vicuna_norm
print(model)

total=0.

for name, param in model.named_parameters():
    total += param.nelement()
print('  + Number of trainable params: %.2fM' % (total / 1e6))
model.save_pretrained('./checkpoints/llava-hr-7b-pretrain-1024-stage2-v2-vicuna')