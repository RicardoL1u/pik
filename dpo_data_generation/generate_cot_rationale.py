
import glob
import json
import pik.utils.openai_proxy as openai_proxy
import re
from tqdm import tqdm
from pik.models.model import Model
import os
from pathlib import Path
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Generate COT rationales')
parser.add_argument('--model_checkpoint', type=str, default='/workspace/MODELS/Qwen1.5-72B-chat', help='Model checkpoint')
parser.add_argument('--prompt_dir', type=str, default='data/bbh/cot-prompts', help='Directory for COT prompts')
parser.add_argument('--target_bbh_dir', type=str, default='data/bbh-new', help='Target directory for BBH data')
parser.add_argument('--debug', action='store_true', help='Debug mode')
args = parser.parse_args()

model_checkpoint = args.model_checkpoint
model_name = Path(model_checkpoint).name
target_bbh_dir = args.target_bbh_dir
# output dir would be data/bbh-new/{MODEL_NAME}
output_dir = os.path.join(target_bbh_dir, model_name)


if not os.path.exists(output_dir):
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir)

print(f"Model checkpoint: {model_checkpoint}")
print(f"Model name: {model_name}")
print(f"Target BBH directory: {target_bbh_dir}")
print(f"Output directory: {output_dir}")



# Load the model
llm = Model(
    model_checkpoint=args.model_checkpoint,
    generation_options= {
        "max_new_tokens": 512,
        "temperature": 1.0,
    },
    is_low_memory = False,
    is_chat_model = True
)


training_files = glob.glob(os.path.join(target_bbh_dir, '*.json'))

training_files = [f for f in training_files if not f.endswith('_prompt.json') and not f.endswith('_result.json')]


task2accuracy = {}

for file in training_files:
    # Task:
    task = file.split('/')[-1].split('.')[0]
    print("Task: ", task)
    print("File: ", file)
    dataset = json.load(open(file))
    if "examples" in dataset:
        dataset = dataset["examples"]
    # load the cot prompt from args.prompt_dir
    cot_prompt = open(os.path.join(args.prompt_dir, f'{task}.txt')).read()
    
    prompt_list = []
    result_list = []
    
    # check if the result file exists
    do_generate = True
    if os.path.exists(os.path.join(output_dir, f'{task}.json')):
        print(f"Do not generate for {task}")
        do_generate = False
        dataset = json.load(open(os.path.join(output_dir, f'{task}.json')))
        if "example" in dataset:
            dataset = dataset["example"]
    
    if args.debug:
        dataset = dataset[10:20]
    
    for data in tqdm(dataset, desc=f"Processing {file}",ncols=100):
        input_prompt = cot_prompt + '\n\nQ: ' + data['input'] + '\nA: Let\'s think step by step.'
        # print(input_prompt)
        # rationales = openai_proxy.chat_completion_use_cache(input_prompt, temperature=1, n=3)
        if do_generate:
            rationales = llm.get_text_generation(input_prompt)
        else:
            rationales = [r['rationale'] for r in data['rationale']]

        
        # extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", model_answer)
        # answers = [rationale.split('.')[-1] for rationale in rationales]
        answers = []
        for rationale in rationales:
            # remove possible \n and . in the end of the rationale
            rationale = re.sub(r'[\n.]+$', '', rationale)
            
            answer_split_dot = rationale.split('.')
            answer_split_new_line = rationale.split('\n\n')
            
            # use the one with the least number of words in the last sentence
            if len(answer_split_dot[-1].split()) < len(answer_split_new_line[-1].split()):
                answer = answer_split_dot[-1]
            else:
                answer = answer_split_new_line[-1]
            answers.append(answer)
            
        is_correct_list = [data['target'] in answer for answer in answers]
        data['rationale'] = [
            {
                'rationale': rationale,
                'answer': answer,
                'is_correct': is_correct
            }
            for rationale, answer, is_correct in zip(rationales, answers, is_correct_list)
        ]
        
    task2accuracy[task] = sum([r['rationale'][0]['is_correct'] for r in dataset]) / len(dataset)
        
    with open(os.path.join(output_dir, f'{task}.json'), 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


# convert task2accuracy to pandas dataframe
df = pd.DataFrame(task2accuracy.items(), columns=['task', 'accuracy'])
# convert to percentage then round to 2 decimal places
df['accuracy'] = (df['accuracy'] * 100).round(2)
print("The accuracy of the model on each task:")
print(df)

# save to csv
df.to_csv(os.path.join(output_dir, 'accuracy.csv'), index=False)

