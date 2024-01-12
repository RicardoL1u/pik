
'''
Generate and save hidden states dataset.
'''
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from pik.models.model import Model
from pik.utils import prompt_eng, evaluate_answer
from typing import List
import logging
import os

# Function Definitions
def load_data(file_path):
    data = pd.read_csv(file_path)
    return [(q, a) for q, a in zip(data.question, data.answer)]

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
    return model.get_batch_hidden_states(text_inputs,
                                         keep_all=args.keep_all_hidden_layers)

def generate_answers(model:Model, text_inputs:List[str]):
    return model.get_batch_text_generation(text_inputs)

def evaluate_model_answers(dataset, output_text_list):
    results = []
    for idx, model_answers in tqdm(enumerate(output_text_list), 
                                   desc='Evaluating Answers', 
                                   total=len(output_text_list)):
        question, correct_answer = dataset[idx]
        for n, model_answer in enumerate(model_answers):
            evaluation = evaluate_answer(model_answer, correct_answer)
            results.append({
                'hid': idx, # hid is same as qid
                'qid': idx,
                'n': n,
                'question': question,
                'answer': correct_answer,
                'model_answer': model_answer,
                'evaluation': evaluation,
            })
    return pd.DataFrame(results)

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_questions', type=int, default=0,
                         help='number of q-a pairs to generate, 0 for all')
    parser.add_argument('--model_checkpoint', '-m', default='MODELS/phi-1.5b',
                         help='model checkpoint to use')
    parser.add_argument('--n_answers_per_question', type=int, default=40,
                         help='number of answers to generate per question')
    parser.add_argument('--max_new_tokens', type=int, default=16,
                         help='maximum number of tokens to generate per answer')
    parser.add_argument('--temperature', type=float, default=1,
                         help='temperature for generation')
    parser.add_argument('--pad_token_id', type=int, default=50256,
                         help='pad token id for generation')
    parser.add_argument('--keep_all_hidden_layers', action='store_true', default=True,
                         help='set to False to keep only hidden states of the last layer')
    parser.add_argument('--hidden_states_filename', default='hidden_states.pt',
                         help='filename for saving hidden states')
    parser.add_argument('--text_generations_filename', default='text_generations.csv',
                         help='filename for saving text generations')
    parser.add_argument('--debug', action='store_true', default=False,
                         help='set to True to enable debug mode')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(
        format="[generate:%(filename)s:L%(lineno)d] %(levelname)-6s %(message)s",
        level=logging.INFO if not args.debug else logging.DEBUG
    )
    # Load data
    try:
        dataset = load_data('data/trivia_qa/val_qa_pairs.csv')
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}")
        exit(1)

    # Initialize model
    model = setup_model(args)

    # Generate hidden states
    text_inputs = [prompt_eng(q, 10, dataset) for q, _ in dataset]
    # Optionally limit number of questions
    if args.n_questions > 0:
        text_inputs = text_inputs[:args.n_questions]
    
    if args.debug:
        text_inputs = text_inputs[:100]
        logging.debug('One Example of text_inputs:\n=======\n%s\n======', text_inputs[0])
        args.hidden_states_filename = args.hidden_states_filename.replace('.pt', '_debug.pt')
        args.text_generations_filename = args.text_generations_filename.replace('.csv', '_debug.csv')
        
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
        results = evaluate_model_answers(dataset, output_text_list)
        results.to_csv(args.text_generations_filename, index=False)
        
    
    