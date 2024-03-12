'''
Generate prompt completions for a list of text inputs for dpo
'''

import json
import glob
import argparse
# import pandas as pd
import torch
from tqdm import tqdm
from pik.models.model import Model
from typing import List
import logging
from transformers import HfArgumentParser
import os



from pik.tools.generate import (
    setup_model as setup_llm_model,
    get_hidden_states, 
    # GenerateArguments
) 
from evaluate import (
    # ProbingArguments, 
    setup_probe_model,
    parse_layers
)


from pik.utils.arguments import (
    GenerateArguments,
    ProbingArguments,
    ScriptArguments
)

if __name__ == "__main__":
    
    parser = HfArgumentParser((GenerateArguments, ProbingArguments, ScriptArguments))
    gen_args, probe_args, script_args = parser.parse_args_into_dataclasses()
    probe_args.model_layer_idx = parse_layers(probe_args.model_layer_idx)
    probe_args.mlp = script_args.mlp
    probe_args.device = script_args.device
    gen_args.mlp = False # we use the hidden states of the last layer of last token
    
    
    logging.basicConfig(
        format="[generate:%(filename)s:L%(lineno)d] %(levelname)-6s %(message)s",
        level=logging.INFO if not script_args.debug else logging.DEBUG
    )
    
    
    # Initialize Probe Model
    probing_model = setup_probe_model(probe_args)
    
    # Initialize language_model
    language_model = setup_llm_model(gen_args)
    
    task2instr = json.load(open("data/bbh_dpo/template/task2instr.json"))
    template = open("data/bbh_dpo/template/template.txt").read()
    training_files = glob.glob('data/bbh_dpo/*.json')
    
    for training_file in training_files:
        task = training_file.split('/')[-1].split('.')[0].replace("_rationale","")
        print("Processing task:", task)
        training_dataset = json.load(open(training_file))
        # open the real test dataset
        test_dataset = json.load(open(f"data/bbh/bbh/{task}.json"))
        # use training dataset as example in the prompt
        for idx, example in tqdm(enumerate(training_dataset), total=len(training_dataset)):
            for rationale_style, rationale in example['rationale'].items():
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
                
                logging.debug('Idx %s Example of prompt_list:\n=======\n%s\n======', idx, prompt_list[idx])
                
                logging.info('Generating hidden states for %d questions', len(prompt_list))
                # Generate hidden states (len(prompt_list), layer_numbs, hidden_size)
                hidden_states = get_hidden_states(language_model, prompt_list, gen_args) 
                logging.info(f"Hidden states shape: {hidden_states.shape}")
                
                # only keep last 4 layers
                hidden_states = hidden_states[:, -4:, :]
                # reshape to (len(prompt_list), hidden_size * layer_numbs)
                hidden_states = hidden_states.view(hidden_states.shape[0], -1)
                
                # convert hidden states to tensor to the same dtype as the probing model
                hidden_states = torch.tensor(hidden_states, dtype=probing_model.fc1.weight.dtype, device=probing_model.fc1.weight.device)               
                # use probing model to predict consistency
                with torch.no_grad():
                    preds: torch.Tensor = probing_model(hidden_states).detach().cpu().squeeze()
                    
                # use the average of the predictions as the final prediction
                final_pred = preds.mean()
                logging.info("Mean prediction: {}".format(final_pred))
                example['rationale'][rationale_style] = {
                    "consistency": final_pred.item(),
                    "rationale": rationale
                }
        
        with open(f"data/bbh_dpo_test/{task}_rationale.json", "w") as f:
            json.dump(training_dataset, f, indent=4, ensure_ascii=False)        