import argparse
import json
import torch

from llava_hr.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_hr.conversation import conv_templates, SeparatorStyle
from llava_hr.model.builder import load_pretrained_model
from llava_hr.utils import disable_torch_init
from llava_hr.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from tqdm import tqdm
import os

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

import re

def find_first_number_group(s):
    # 使用正则表达式匹配第一个以非数字字符结尾的数字组
    match = re.search(r'\d+(?=\D|$)', s)
    if match:
        # 如果找到匹配项，将其转换为整数
        return int(match.group())
    else:
        # 如果没有找到匹配项，返回 None 或其他默认值
        return None

def main(args):
    # Model
    disable_torch_init()
    # args.model_path = args.model_base
    # args.model_base = None

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.eval()

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    total_error = 0
    right_num = 0
    total_num = 0
    to_save = []
    with open(args.test_file) as f:
        data = json.load(f)
    for i, item in tqdm(enumerate(data)):
        
        
        conv = conv_templates[args.conv_mode].copy()
        image_path = os.path.join(args.image_folder, item["image"])
        image = load_image(image_path)
        image_size = image.size
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        inp = item["conversations"][0]["value"]

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None
        
        conv.append_message(item["conversations"][0]["from"], inp)
        conv.append_message(item["conversations"][1]["from"], None)
        prompt = conv.get_prompt()
        # prompt = prompt[:-4]+" Note that you should regard the top of object as its value. gpt: "
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)


        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        conv.messages[-1][-1] = outputs
        pred = find_first_number_group(outputs)
        item["predict"] = outputs
        to_save.append(item)
        if item["type"] == "extreme value":
            continue
        if pred is None:
            pred = 0
        gt = int(item["conversations"][1]["value"])
        error = abs(gt-pred)
        if error < 20:
            right_num += 1
        total_num += 1
        print('acc', right_num / total_num)
        # total_error += error
        # print('ave_error:', total_error // (i+1))
        if args.debug:
            print("\n", {"prompt": prompt}, "\n")
            print("outputs:", outputs, "\n")
            print("ground truth:", gt)
            input("按 Enter 继续...")
        
    with open(args.save_name, "w") as f:
        json.dump(to_save, f, indent=4)       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='/mnt/share/xujing/LLaVA-HR-Lora/checkpoints/llava-mllm-dataset3-lorabackbone')
    parser.add_argument("--model-base", type=str, default="/mnt/share/xujing/checkpoints/ChartMLLM")
    parser.add_argument("--test-file", type=str, default="/mnt/share/xujing/flux/val_dataset_3.json")
    parser.add_argument("--image-folder", type=str, default="/mnt/share/xujing/flux/version3")
    parser.add_argument("--save-name", default="test_chartmllm-data3-tuned-lorabackbone.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.model_base == "None":
        args.model_base = None
    main(args)
