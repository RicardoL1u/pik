'''
Generate and save hidden states dataset.
'''
import argparse
import os
import pandas as pd
import torch
from IPython.display import display
from time import time
from tqdm import trange,tqdm
from pik.datasets.triviaqa_dataset import TriviaQADataset
from pik.models.model import Model
from pik.utils import prompt_eng, evaluate_answer
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from vllm import LLM
from pik.models.model import load_model
import logging
import json
logging.basicConfig(
    format="[generate:%(filename)s:L%(lineno)d] %(levelname)-6s %(message)s",
    level=logging.DEBUG
)


# Set params
parser = argparse.ArgumentParser()
parser.add_argument('--n_questions', type=int, default=10, help='number of q-a pairs to generate, index of selected questions is saved in `data_ids`; if <= 0, all questions are used')
parser.add_argument('--dataset_seed', type=int, default=420, help='seed for selecting questions from dataset')
parser.add_argument('--generation_seed', type=int, default=1337, help='seed for generation reproducibility')
parser.add_argument('--model_checkpoint', '-m', default='EleutherAI/gpt-j-6B', help='model checkpoint to use')
parser.add_argument('--precision', '-p', default='float16', help='model precision')
parser.add_argument('--n_answers_per_question', type=int, default=40, help='number of answers to generate per question')
parser.add_argument('--max_new_tokens', type=int, default=16, help='maximum number of tokens to generate per answer')
parser.add_argument('--temperature', type=float, default=1, help='temperature for generation')
parser.add_argument('--pad_token_id', type=int, default=50256, help='pad token id for generation')
parser.add_argument('--keep_all_hidden_layers', action='store_true', default=True, help='set to False to keep only hidden states of the last layer')
parser.add_argument('--save_frequency', type=int, default=99999, help='write results to disk after every `save_frequency` questions')
parser.add_argument('--data_folder', default='data', help='data folder')
parser.add_argument('--hidden_states_filename', default='hidden_states.pt', help='filename for saving hidden states')
parser.add_argument('--text_generations_filename', default='text_generations.csv', help='filename for saving text generations')
parser.add_argument('--qa_pairs_filename', default='qa_pairs.csv', help='filename for saving q-a pairs')
parser.add_argument('--debug', action='store_true', default=False, help='set to True to enable debug mode')
# parser.add_argument('--estimate', action='store_true', default=False, help='set to True to estimate time to completion')
# parser.add_argument('--n_test', type=int, default=3, help='number of questions to use for estimating time to completion')
args = parser.parse_args()
args.precision = torch.float16 if args.precision == 'float16' else torch.float32
torch.set_default_dtype(args.precision)
args.hidden_states_filename = os.path.join(args.data_folder, args.hidden_states_filename)
args.text_generations_filename = os.path.join(args.data_folder, args.text_generations_filename)
args.qa_pairs_filename = os.path.join(args.data_folder, args.qa_pairs_filename)

generation_options = {
    'max_new_tokens': args.max_new_tokens,
    'temperature': args.temperature,
    'do_sample': True,
    'pad_token_id': args.pad_token_id,
    'n': args.n_answers_per_question,
    # 'eos_token_id': 198,	# Stop generating more tokens when the model generates '\n'
    # 'eos_token_id': 13,	# Stop generating more tokens when the model generates '.'
}

# Load dataset and model
data = TriviaQADataset()
# torch.manual_seed(args.dataset_seed)
# data_ids = torch.randperm(len(data))
# if args.n_questions == 0:
#     args.n_questions = len(data_ids)
# if args.n_questions > 0:
# 	data_ids = data_ids[:args.n_questions]
# data_ids = data_ids.cpu().numpy().tolist()
model = Model(args.model_checkpoint, precision=args.precision,generation_options=generation_options)


start = time()
all_hidden_states = None
# torch.manual_seed(args.generation_seed)

text_input_list = []
for question, _ in data:
    text_input_list.append(prompt_eng(question, 10, data))

if args.debug:
    text_input_list = text_input_list[:1000]
    logging.debug('text_input_list: %s', text_input_list[:2])

    


if os.path.exists(args.hidden_states_filename):
    hidden_states_host = torch.load(args.hidden_states_filename)
    logging.info('Loaded hidden states from disk')
else:
    if model.mode == 'lazy':
        model.model = load_model(args.model_checkpoint, is_vllm=False)

    hidden_states = None
    hidden_states_host = None
    for idx, text_input in tqdm(enumerate(text_input_list),desc='Pre-generating hidden states', total=len(text_input_list)):
        hidden_state = model.get_hidden_states(text_input, keep_all=args.keep_all_hidden_layers)
        if hidden_states is None:
            hidden_states = hidden_state.unsqueeze(0)
        else:
            hidden_states = torch.cat((hidden_states, hidden_state.unsqueeze(0)), dim=0)
        # periodically move hidden states to cpu
        if hidden_states.shape[0] % 100 == 0 or idx == len(text_input_list) - 1:
            if hidden_states_host is None:
                hidden_states_host = hidden_states.cpu()
            else:
                hidden_states_host = torch.cat((hidden_states_host, hidden_states.cpu()), dim=0)
            hidden_states = None

    torch.save(hidden_states_host, args.hidden_states_filename)


if model.mode == 'lazy':
    # release memory
    model.model = None
    model.vllm_model = load_model(args.model_checkpoint, is_vllm=True)


# use batch generation
output_text_list = model.get_batch_text_generation(text_input_list)

if args.debug:
    with open(f'{args.data_folder}/output_text_list.json', 'w') as f:
        json.dump(output_text_list, f, indent=4, ensure_ascii=False)

### Start generating
results = [] 
for idx, model_answers in tqdm(enumerate(output_text_list),
                               desc='Evaluate Answers',
                               total=len(output_text_list)):
    # model_answers = model.get_text_generation(text_input)
    qusetion, answer = data[idx]
    for n, model_answer in enumerate(model_answers):
        eval = evaluate_answer(model_answer, answer)
        # Record results in memory
        results.append({
            'hid': idx, # hid is same as qid
            'qid': idx,
            'n': n,
            'question': qusetion,
            'answer': answer,
            'model_answer': model_answer,
            'evaluation': eval
        })
# convert to dataframe
results = pd.DataFrame(results)


# progress_bar = trange(args.n_questions) if not args.estimate else trange(args.n_test)
# for i in progress_bar:
# 	# Prep inputs
# 	hid, qid = i, data_ids[i]
# 	question, answer = data[qid]
# 	progress_bar.set_description(question)
# 	text_input = prompt_eng(question, 10, data)
# 	# Get hidden state
# 	hidden_state = model.get_hidden_states(text_input, keep_all=args.keep_all_hidden_layers)
# 	if hid == 0:
# 		all_hidden_states = hidden_state.unsqueeze(0)
# 	else:
# 		all_hidden_states = torch.cat((all_hidden_states, hidden_state.unsqueeze(0)), dim=0)
# 	# Generate multiple model answers
# 	model_answers = model.get_text_generation(text_input)
# 	for n, model_answer in enumerate(model_answers):
# 		eval = evaluate_answer(model_answer, answer)
# 		# Record results in memory
# 		df_idx = results.shape[0]
# 		results.loc[df_idx, 'hid'] = hid
# 		results.loc[df_idx, 'qid'] = qid
# 		results.loc[df_idx, 'n'] = n
# 		results.loc[df_idx, 'model_answer'] = model_answer
# 		results.loc[df_idx, 'evaluation'] = eval
# 	# Periodically write results to disk
# 	if not args.estimate and (hid + 1) % args.save_frequency == 0 and (hid + 1) != args.save_frequency:
# 		torch.save(all_hidden_states, args.hidden_states_filename)
# 		for col in results.columns:
# 			if col != 'model_answer':
# 				results[col] = results[col].astype(int)
# 		results.to_csv(args.text_generations_filename, index=False)
# 		logging.info('-------------')
# 		logging.info(results)
# 		logging.info(args.text_generations_filename)

# Estimate time to completion and disk usage if `--estimate` is set, then exit
# if args.estimate:
# 	time_taken = time() - start
# 	num_floats = args.n_questions
# 	for dim in all_hidden_states.shape[1:]:
# 		num_floats *= dim
# 	bytes_per_float = 2 if args.precision == torch.float16 else 4
# 	logging.info(f'''n={args.n_test} questions
# 	{args.n_answers_per_question} generations per question
# 	{generation_options['max_new_tokens']} new tokens per generation
# 	-------------------------------------
# 	Average processing time per question:
# 	{time_taken / args.n_questions :.2f} seconds

# 	Estimated duration (give or take) to process all {args.n_questions} questions:
# 	{(time_taken / args.n_test) * args.n_questions / 3600 :.3f} hours
    
# 	Estimated disk usage for {args.hidden_states_filename}:
# 	{(num_floats * bytes_per_float) / 1e6:.3f} MB''')
# 	exit()

# Generate q-a pairs df
qa_pairs = []
for idx in range(len(data)):
    q, a = data[idx]
    qa_pairs.append({
        'qid': idx,
        'question': q,
        'answer': a
    })
qa_pairs = pd.DataFrame(qa_pairs)


# Generate results df
for col in results.columns:
    if col != 'model_answer':
        results[col] = results[col].astype(int)
display(results.head())
logging.info('Mean evaluation score:', results.evaluation.mean())
logging.info('Hidden states shape:', hidden_states_host.shape)

# Write final results to disk
# torch.save(hidden_states_host, args.hidden_states_filename)
results.to_csv(args.text_generations_filename, index=False)
qa_pairs.to_csv(args.qa_pairs_filename, index=False)
