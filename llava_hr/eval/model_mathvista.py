import os
import io
import time
import argparse

from tqdm import tqdm

import sys
# sys.path.append('../')
from llava_hr.eval.utilities import *


from llava_hr.eval.build_query import create_query_data
from llava_hr.model.builder import load_pretrained_model
from llava_hr.utils import disable_torch_init
from llava_hr.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import torch
import shortuuid

from llava_hr.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_hr.conversation import conv_templates, SeparatorStyle
def verify_response(response):
    if isinstance(response, str):
        response = response.strip() 
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True


def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout
    
    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e
    
    # Restore the original stdout
    sys.stdout = old_stdout
    
    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()
    
    # Return the captured output or error
    return captured_output, error
    
def get_response(model,image_path, query,args):
    # idx = line["question_id"]
    image_file = image_path
    qs = query
    cur_prompt = qs
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs+'\n'+'Answer the question using a single word or phrase.'
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs +'\n'+'Answer the question using a single word or phrase.'

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image = Image.open(image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    pattern = r"(.)\1+$"
    outputs = re.sub(pattern, r"\1", outputs)
    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--data_dir', type=str, default='./playground/data/eval/mathvista')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='./playground/data/eval/mathvista/results/bard')
    parser.add_argument('--output_file', type=str, default='output_bard.json')
    # model
    parser.add_argument('--model', type=str, default='llava', help='llm engine',
                        choices = ['gpt-3.5-turbo', 'claude-2', 'gpt4', 'gpt-4-0613', 'bard','llava'])
    parser.add_argument('--key', type=str, default='', help='key for llm api')
    # query
    parser.add_argument('--query_file', type=str, default=None)  
    parser.add_argument('--caption_file', type=str, default='./playground/data/eval/mathvista/texts/captions_bard.json')
    parser.add_argument('--ocr_file', type=str, default='./playground/data/eval/mathvista/texts/ocrs_easyocr.json')
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type', 
                        choices = ['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', action='store_true', help='use caption data')
    parser.add_argument('--use_ocr', action='store_true', help='use ocr data')
    # other settings
    parser.add_argument('--rerun', action='store_true', help='rerun answer extraction for all problems')
    parser.add_argument('--debug', action='store_true', help='debug mode')



    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    # load data
    input_file = os.path.join(args.data_dir, args.input_file)
    print(f"Reading {input_file}...")
    data = read_json(input_file)
    
    # load or create query data
    if args.query_file:
        query_file = os.path.join(args.data_dir, args.query_file)
        if os.path.exists(query_file):
            print(f"Loading existing {query_file}...")
            query_data = read_json(query_file)
    else:
        print("\nCreating new query...")
        # load caption
        caption_data = {}
        if args.use_caption:
            caption_file = args.caption_file
            if os.path.exists(caption_file):
                print(f"Reading {caption_file}...")
                try:
                    caption_data = read_json(caption_file)["texts"]
                    print("Caption data loaded.")
                except:
                    print("Caption data not found!! Please Check.")                    
        # load ocr
        ocr_data = {}
        if args.use_ocr:
            ocr_file = args.ocr_file
            if os.path.exists(ocr_file):
                print(f"Reading {ocr_file}...")
                try:
                    ocr_data = read_json(ocr_file)["texts"]
                    print("OCR data loaded.")
                except:
                    print("OCR data not found!! Please Check.")
        # create query
        query_data = create_query_data(data, caption_data, ocr_data, args)

    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)
    
    # load results
    if os.path.exists(output_file):
        print("\nResults already exist.")
        print(f"Reading {output_file}...")
        results = read_json(output_file)
    else:
        results = {}

    # load model
    print(f"\nLoading {args.model}...")
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    print(f"Model loaded.")
    
    # build final test pid list
    test_pids = list(data.keys())
    print("\nNumber of test problems in total:", len(test_pids))

    skip_pids = []
    if not args.rerun:
        print("\nRemoving problems with existing valid response...")
        for pid in test_pids:
            # print(f"Checking {pid}...")
            if pid in results and 'response' in results[pid]:
                response = results[pid]['response']
                if verify_response(response):
                    # print(f"Valid response found for {pid}.")
                    skip_pids.append(pid)
    else:
        print("\nRerun answer extraction for all problems...")

    test_pids = [pid for pid in test_pids if pid not in skip_pids]
    print("Number of test problems to run:", len(test_pids))
    # print(test_pids)

    # tqdm, enumerate results
    for _, pid in enumerate(tqdm(test_pids)):
        problem = data[pid]
        query = query_data[pid]
        image = problem['image']
        image_path = os.path.join(args.data_dir, image)

        if args.debug:
            print("--------------------------------------------------------------")
        print(f"\nGenerating response for {pid}...")
        try:
            response = get_response(model,image_path, query,args)
            # print(f"Response: {response}")
            results[pid] = problem
            results[pid]['query'] = query
            if args.shot_type == 'solution':
                results[pid]['response'] = response
            else:
                output, error = evaluate_code(response)
                results[pid]['response'] = response
                results[pid]['execution'] = output
                results[pid]['error'] = str(error)
            if args.debug:
                print(f"\n#Query: \n{query}")
                print(f"\n#Response: \n{response}")
        except Exception as e:
            print(e)
            print(f"Error in extracting answer for {pid}")
            results[pid]['error'] = e
    
        try:
            print(f"Saving results to {output_file}...")
            save_json(results, output_file)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")
