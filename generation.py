
'''
Generate and save hidden states dataset.
'''
import argparse
# import pandas as pd
import torch
from tqdm import tqdm
from pik.models.model import Model
from pik.utils.utils import (
    prompt_eng_for_freeqa,
    prompt_eng_for_mcq,
    load_dataset,
    load_template,
    load_example,
)
from pik.utils.special_cases import check_special_cases
from typing import List
import logging
import os
import json

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
    for batch_texts in tqdm(range(0, len(text_inputs), 32*10), 
                            desc='Generating hidden states',
                            total=len(text_inputs)//(32*10),
                            ncols=100):
        batch_texts = text_inputs[batch_texts:batch_texts+32*10]
        if args.mlp:
            batch_hidden_states = model.get_batch_MLP_activations(batch_texts, keep_all=args.keep_all_hidden_layers)
        else:
            batch_hidden_states = model.get_batch_hidden_states(batch_texts, keep_all=args.keep_all_hidden_layers)
        full_hidden_states.append(batch_hidden_states)
    return torch.cat(full_hidden_states, dim=0)

def generate_answers(model:Model, text_inputs:List[str]):
    return model.get_batch_text_generation(text_inputs)

def evaluate_model_answers(dataset, output_text_list, evaluate_answer:callable):
    results = []
    for idx, model_answers in tqdm(enumerate(output_text_list), 
                                   desc='Evaluating Answers', 
                                   total=len(output_text_list)):
        
        question, correct_answer = dataset[idx]['question'], dataset[idx]['answer']
        results.append({
            'qid': idx,
            'question': question,
            'answer': correct_answer,
            'model_answers': model_answers,
            'unit_evaluations': [evaluate_answer(model_answer, correct_answer) for model_answer in model_answers],
        })
        results[-1]['evaluation'] = sum(results[-1]['unit_evaluations'])/len(results[-1]['unit_evaluations'])
    return results

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--mlp', action='store_true', default=False,
                         help='set to True to use MLP activation hook')
    parser.add_argument('--dataset', default='trivia_qa',
                            help='dataset to use')
    parser.add_argument('--example_file', default='data/trivia_qa/trivia_qa_examples.json',
                            help='example file to use')
    parser.add_argument('--template', default='icl',
                            help='template to use')
    parser.add_argument('--shot', type=int, default=4,
                         help='number of examples per question')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(
        format="[generate:%(filename)s:L%(lineno)d] %(levelname)-6s %(message)s",
        level=logging.INFO if not args.debug else logging.DEBUG
    )
    
    # Checks Special Cases
    check_special_cases(args)
    
    # Load data
    try:
        dataset, evaluate_answer = load_dataset(args.dataset)
        logging.info(f'Loaded {len(dataset)} instance data from {args.dataset}')
        examples = load_example(args.example_file)
        logging.info(f'Loaded {len(examples)} example data from {args.example_file}')
        template:str = load_template(args.template)
        logging.info(f"Evaluating answers using {evaluate_answer.__name__}")
        logging.info(f'Using template: {args.template}')
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}")
        exit(1)


    # Prepare for evaluation
    if args.dataset == 'commonsense_qa':
        logging.info('Preparing for evaluation of commonsense_qa')
        new_dataset = []
        for data in dataset:
            new_data = data.copy()  # Create a copy of the original dictionary
            new_data['answer'] = {
                "choices": data["choices"],
                "answerKey": data["answerKey"]
            }
            new_dataset.append(new_data)

        dataset = new_dataset
            
    assert 'answer' in dataset[0], 'Dataset must have "answer" field'
    
    prompt_eng = prompt_eng_for_mcq if args.template == 'mcq_cmqa' else prompt_eng_for_freeqa
    

    text_inputs = [prompt_eng(sample, examples, template, args.shot) 
                   for sample in dataset]


    
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
        results:dict = evaluate_model_answers(dataset, output_text_list, evaluate_answer)
        logging.info(f'The mean of evaluation is {sum([result["evaluation"] for result in results])/len(results)}')
        with open(args.text_generations_filename, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    
    