import json
import glob
import argparse
import torch
import random
random.seed(42)
randomx_index_list = random.sample(range(0, 100), 20)
from tqdm import tqdm
import numpy as np
from pik.models.model import Model
from .generate_cot_rationale import check_response, FULL_ALPHABET_OPTIONS
from typing import List, Tuple
import logging
from transformers import HfArgumentParser
import os

import time
import random
from pik.tools.generate import (
    setup_model as setup_llm_model,
    get_hidden_states,
    generate_answers
)
from evaluate import (
    setup_probe_model,
    parse_layers
)

from pik.utils.arguments import (
    GenerateArguments,
    ProbingArguments,
    ScriptArguments
)

TASK2INSTR = json.load(open("temp_data/bbh/cot/task2instr.json"))
TEMPLATE = open("temp_data/bbh/cot/template.txt").read()

def process_task(task, example_dir, training_file, test_dataset, language_model, probing_model, gen_args, probe_args):
    training_dataset = json.load(open(training_file))

    
    output_path = f"{example_dir}/rating/{task}_rationale.json"
    if os.path.exists(output_path):
        logging.info(f"Skipping {task} as it already exists")
        return
    
    test_dataset['examples'] = [test_dataset['examples'][i] for i in randomx_index_list]
    
    total_detail_preds = []
    for idx, example in tqdm(enumerate(training_dataset), total=len(training_dataset)):
        if True:
            example, detail_preds = process_unit_example_by_decoding(task, example, test_dataset, language_model)
        else:
            example, detail_preds = process_unit_example_by_hidden_states(task, example, test_dataset, language_model, probing_model, gen_args, probe_args)
        total_detail_preds.append(detail_preds)
        training_dataset[idx] = example

    # total_detail_preds is list[Tensor(n_rationale, n_test_examples)]
    # convert it to a full tensor and reshape it as (n_examples, n_rationales, n_test_examples)
    total_detail_preds = torch.cat(total_detail_preds, dim=0)
    total_detail_preds = total_detail_preds.view(len(training_dataset), len(example['rationale']), -1)
    # save the detailed predictions
    torch.save(total_detail_preds, f"{example_dir}/rating/{task}_detailed_preds.pt")
    
    with open(output_path,"w") as f:
        json.dump(training_dataset, f, indent=4, ensure_ascii=False)

def process_unit_example_by_decoding(task, example, test_dataset, language_model):
    prompt_list = []
    
    detail_acc_list = []
    for rationale in example['rationale']:
        prompt_list = generate_prompt_list(task, example, rationale['rationale'], test_dataset)
        response_list_of_list = generate_answers(language_model, prompt_list)
        is_correct_list = []
        for response_list, test_ex in zip(response_list_of_list, test_dataset['examples']):
            unit_question_ave = sum([check_response(response, test_ex['input'], test_ex['target'], task) for response in response_list]) / len(response_list)
            is_correct_list.append(unit_question_ave)
        rationale['accuracy'] = sum(is_correct_list) / len(is_correct_list)
        detail_acc_list.append(is_correct_list)
    
    # convert to torch tensor
    detail_acc_list = torch.tensor(detail_acc_list, dtype=torch.float32)
    return example, detail_acc_list

def process_unit_example_by_hidden_states(task, example, test_dataset, language_model, probing_model, gen_args, probe_args):
    prompt_list = []
    
    detail_consistency_list = []
    for rationale in example['rationale']:
        prompt_list = generate_prompt_list(task, example, rationale['rationale'], test_dataset)
        hidden_states = get_and_process_hidden_states(language_model, prompt_list, gen_args, probe_args)
        preds_mean, preds = predict_consistency(probing_model, hidden_states)
        rationale['consistency'] = preds_mean.item()
        detail_consistency_list.append(preds)
    
    # convert to torch tensor
    return example, torch.stack(detail_consistency_list)

def generate_prompt_list(task, example, rationale, test_dataset):
    prompt_list = []
    for to_test_data in test_dataset['examples']:
        prompt = TEMPLATE.format(
            instr=TASK2INSTR[task],
            example_input=example['input'],
            example_rationale=rationale,
            example_target=example['target'],
            input=to_test_data['input']
        )
        prompt_list.append(prompt)
    return prompt_list

def get_and_process_hidden_states(language_model, prompt_list, gen_args, probe_args):
    # temporarliy turn off tqdm bar
    
    hidden_states = get_hidden_states(language_model, prompt_list, gen_args)
    hidden_states = hidden_states[:, -4:, :]
    hidden_states = hidden_states.view(hidden_states.shape[0], -1)
    hidden_states = torch.tensor(hidden_states, dtype=torch.float32, device=probe_args.device)
    return hidden_states

def predict_consistency(probing_model, hidden_states) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        preds = probing_model(hidden_states).detach().cpu().squeeze()
    return preds.mean(), preds


if __name__ == "__main__":
    parser = HfArgumentParser((GenerateArguments, ProbingArguments, ScriptArguments))
    gen_args, probe_args, script_args = parser.parse_args_into_dataclasses()
    probe_args.model_layer_idx = parse_layers(probe_args.model_layer_idx)
    probe_args.mlp = script_args.mlp
    probe_args.device = script_args.device
    gen_args.mlp = False

    example_dir = script_args.example_dir
    
    logging.basicConfig(
        format="[generate:%(filename)s:L%(lineno)d] %(levelname)-6s %(message)s",
        level=logging.INFO if not script_args.debug else logging.DEBUG
    )

    logging.info("Example directory: %s", example_dir)

    probing_model = setup_probe_model(probe_args)
    language_model = setup_llm_model(gen_args)
    
    language_model.is_low_memory = False
    
    training_files = glob.glob(f'{example_dir}/*.json')
    training_files = [f for f in training_files if "_rationale" not in f and '_result' not in f]
    
    
    random.seed(time.time())
    random.shuffle(training_files)
    
    for training_file in training_files:
        task = training_file.split('/')[-1].split('.')[0]
        print("Processing task:", task)
        test_dataset = json.load(open(f"temp_data/bbh/cot/bbh-acl/files/{task}.json"))
        process_task(task, example_dir, training_file, test_dataset, language_model, probing_model, gen_args, probe_args)
