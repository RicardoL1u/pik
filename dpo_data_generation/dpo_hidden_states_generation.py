import json
import glob
import argparse
import torch
from tqdm import tqdm

from pik.models.model import Model
from typing import List, Tuple
import logging
from transformers import HfArgumentParser
import os

import time
import random
from pik.tools.generate import (
    setup_model as setup_llm_model,
    get_hidden_states,
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

def process_task(task, training_file, test_dataset, language_model, probing_model, gen_args, probe_args):
    training_dataset = json.load(open(training_file))
    task2instr = json.load(open("temp_data/bbh/cot/task2instr.json"))
    template = open("temp_data/bbh/cot/template.txt").read()

    
    output_path = f"temp_data/bbh/cot/bbh-new/gpt-3.5-turbo/cot-prompts-3-shot/rating/{task}_rationale.json"
    if os.path.exists(output_path):
        logging.info(f"Skipping {task} as it already exists")
        return
        
    total_detail_preds = []
    for idx, example in tqdm(enumerate(training_dataset), total=len(training_dataset)):
        prompt_list = []
        hidden_states = None
        
        # rationale_styles, rationale_list = [], []
        rationale_list = []
        for rationale in example['rationale']:
            # rationale_styles.append(rationale_style)
            rationale_list.append(rationale['rationale'])
        
        for rationale in rationale_list:
            prompt_list += generate_prompt_list(task, example, rationale, test_dataset, task2instr, template)

        hidden_states = get_and_process_hidden_states(language_model, prompt_list, gen_args, probe_args)
        logging.info("==== Hidden states ====")
        logging.info(f"Hidden states shape: {hidden_states.shape}")
        
        step = len(test_dataset['examples'])
        
        final_preds, detailed_preds = [], []
        for i in range(0, len(hidden_states), step):
            preds_mean, preds = predict_consistency(probing_model, hidden_states[i:i+step])
            final_preds.append(preds_mean)
            detailed_preds.append(preds)
        detailed_preds = torch.cat(detailed_preds, dim=0)
        total_detail_preds.append(detailed_preds)
        
        logging.info("==== Final Predictions ====")
        logging.info(final_preds)
        for rationale_unit, final_pred in zip(example['rationale'], final_preds):
            rationale_unit['consistency'] = final_pred.item()

    # total_detail_preds is list[Tensor(n_rationale, n_test_examples)]
    # convert it to a full tensor and reshape it as (n_examples, n_rationales, n_test_examples)
    total_detail_preds = torch.cat(total_detail_preds, dim=0)
    total_detail_preds = total_detail_preds.view(len(training_dataset), len(example['rationale']), -1)
    # save the detailed predictions
    torch.save(total_detail_preds, f"temp_data/bbh/cot/bbh-new/gpt-3.5-turbo/cot-prompts-3-shot/rating/{task}_detailed_preds.pt")
    
    with open(output_path,"w") as f:
        json.dump(training_dataset, f, indent=4, ensure_ascii=False)

def generate_prompt_list(task, example, rationale, test_dataset, task2instr, template):
    prompt_list = []
    for to_test_data in test_dataset['examples']:
        prompt = template.format(
            instr=task2instr[task],
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

    logging.basicConfig(
        format="[generate:%(filename)s:L%(lineno)d] %(levelname)-6s %(message)s",
        level=logging.INFO if not script_args.debug else logging.DEBUG
    )

    probing_model = setup_probe_model(probe_args)
    language_model = setup_llm_model(gen_args)

    training_files = glob.glob('temp_data/bbh/cot/bbh-new/gpt-3.5-turbo/cot-prompts-3-shot/*.json')
    training_files = [f for f in training_files if "_rationale" not in f and '_result' not in f]
    
    
    random.seed(time.time())
    random.shuffle(training_files)
    
    for training_file in training_files:
        task = training_file.split('/')[-1].split('.')[0]
        print("Processing task:", task)
        test_dataset = json.load(open(f"temp_data/bbh/cot/bbh-acl/files/{task}.json"))
        process_task(task, training_file, test_dataset, language_model, probing_model, gen_args, probe_args)
