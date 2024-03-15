
'''
Generate and save hidden states dataset.
'''
import argparse
# import pandas as pd
import torch
from tqdm import tqdm
from pik.models.model import Model
from typing import List
import logging
import os
import json

from dataclasses import dataclass, field
from transformers import HfArgumentParser

# Function Definitions
# def load_data(file_path):
#     data = pd.read_csv(file_path)
#     return [(q, a) for q, a in zip(data.question, data.answer)]

def setup_model(args):
    generation_options = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'do_sample': True,
        'pad_token_id': args.pad_token_id,
        'n': args.n_answers_per_question,
    }
    return Model(args.model_checkpoint, generation_options=generation_options)

# def get_hidden_states(model:Model, text_inputs:List[str], args):
#     hidden_states = None
    
#     for text_input in tqdm(text_inputs, desc='Generating hidden states'):
#         state = model.get_hidden_states(text_input, keep_all=args.keep_all_hidden_layers)
        
#         # periodically move hidden states to cpu
#         if hidden_states is None:
#             hidden_states = state.unsqueeze(0).cpu()
#         else:
#             hidden_states = torch.cat((hidden_states, state.unsqueeze(0).cpu()), dim=0)
    
#     # release memory
#     model.model = None
#     return hidden_states

def get_hidden_states(model:Model, text_inputs, args):
    full_hidden_states = []
    for batch_texts in tqdm(range(0, len(text_inputs), 128*10), 
                            desc='Generating hidden states',
                            total=len(text_inputs)//(128*10),
                            ncols=100):
        batch_texts = text_inputs[batch_texts:batch_texts+128*10]
        if args.mlp:
            batch_hidden_states = model.get_batch_MLP_activations(batch_texts, keep_all=args.keep_all_hidden_layers)
        else:
            batch_hidden_states = model.get_batch_hidden_states(batch_texts, keep_all=args.keep_all_hidden_layers)
        full_hidden_states.append(batch_hidden_states)
    return torch.cat(full_hidden_states, dim=0)

def generate_answers(model:Model, text_inputs:List[str]):
    return model.get_batch_text_generation(text_inputs)

# Argument Parsing
@dataclass
class GenerateArguments:
    model_checkpoint: str = field(default='MODELS/phi-1.5b', metadata={"help": "model checkpoint to use"})
    n_answers_per_question: int = field(default=40, metadata={"help": "number of answers to generate per question"})
    max_new_tokens: int = field(default=256, metadata={"help": "maximum number of tokens to generate per answer"})
    temperature: float = field(default=1.0, metadata={"help": "temperature for generation"})
    pad_token_id: int = field(default=50256, metadata={"help": "pad token id for generation"})
    keep_all_hidden_layers: bool = field(default=True, metadata={"help": "set to False to keep only hidden states of the last layer"})
    hidden_states_filename: str = field(default='hidden_states.pt', metadata={"help": "filename for saving hidden states"})
    text_generations_filename: str = field(default='text_generations.csv', metadata={"help": "filename for saving text generations"})
    debug: bool = field(default=False, metadata={"help": "set to True to enable debug mode"})
    mlp: bool = field(default=False, metadata={"help": "set to True to use MLP activation hook"})
    input: str = field(default='data/qa_dataset.json', metadata={"help": "filename for input dataset"})




def parse_arguments():
    parser = HfArgumentParser(GenerateArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    return script_args

if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(
        format="[generate:%(filename)s:L%(lineno)d] %(levelname)-6s %(message)s",
        level=logging.INFO if not args.debug else logging.DEBUG
    )
    
    text_inputs = json.load(open(args.input, 'r'))

    
    if args.debug:
        text_inputs = text_inputs[:1000]
        for idx, text in enumerate(text_inputs):
            logging.debug('Idx %s Example of text_inputs:\n=======\n%s\n======', idx, text)
        args.hidden_states_filename = args.hidden_states_filename.replace('.pt', '_debug.pt')
        args.text_generations_filename = args.text_generations_filename.replace('.json', '_debug.json')
      
    # Initialize model
    model = setup_model(args) 
        
    # Generate hidden states   
    if not os.path.exists(args.hidden_states_filename):
        
        logging.info('Generating hidden states for %d questions', len(text_inputs))
        hidden_states = get_hidden_states(model, text_inputs, args)
        # Save hidden states
        logging.info(f'Saved hidden states to {args.hidden_states_filename}')
        torch.save(hidden_states, args.hidden_states_filename)

    if not os.path.exists(args.text_generations_filename):
        # Generate answers
        output_text_list = generate_answers(model, text_inputs)
        # Save results
        logging.info(f'Saved text generations to {args.text_generations_filename}')
        
        with open(args.text_generations_filename, 'w') as f:
            json.dump(output_text_list, f, indent=4, ensure_ascii=False)
    
    