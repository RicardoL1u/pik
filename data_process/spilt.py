import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--text_generation_results_path', type=str, required=True)
parser.add_argument('--hidden_states_results_path', type=str, required=True)
parser.add_argument('--ood_task_list', type=str, default='data/bbh/cot-prompts-1-shot')
args = parser.parse_args()
text_generation_results_path = args.text_generation_results_path
hidden_states_results_path = args.hidden_states_results_path

assert '/full/' in text_generation_results_path, "Please use the full generation results"
assert '/full/' in hidden_states_results_path, "Please use the full hidden states results"


print(f"Splitting the results in {text_generation_results_path} and {hidden_states_results_path}")


ood_tasks = json.load(open(args.ood_task_list))
print("Out of domain tasks:")
print(ood_tasks)

import glob
task_files = 'data/bbh/cot-prompts-1-shot'
task_files = glob.glob(task_files + '/*.txt')

# model_name = 'gemma-2b'

# # random split into two sets
# import random
# random.seed(42)
# random.shuffle(task_files)
# iid_tasks = task_files[:18]
# ood_tasks = task_files[18:]
# print("Out of domain tasks:")
# print(ood_tasks)

# load the cot prompt of each task
cot_prompts = {
    task_name.split('/')[-1].split('.')[0]: open(task_name).read()
    for task_name in task_files
}

prompt2task = {
    prompt:task.split('/')[-1].split('.')[0]
    for task, prompt in cot_prompts.items()
}

# get ood prompts
ood_prompts = set([cot_prompts[task_name] for task_name in ood_tasks])

text_generation_results = json.load(open(text_generation_results_path))

# find the index of iid generation
# "question" in iid generation should be start with any of the iid prompts
iid_index = set()
ood_index = set()
for i, gen in enumerate(text_generation_results):
    for prompt in cot_prompts.values():
        if gen['question'].startswith(prompt):
            if 'task' not in text_generation_results[i]:
                text_generation_results[i]['task'] = prompt2task[prompt]
            if prompt not in ood_prompts:
                iid_index.add(i)
            else:
                ood_index.add(i)
            
assert len(iid_index) + len(ood_index) == len(text_generation_results), f"iid: {len(iid_index)}, ood: {len(ood_index)}, total: {len(text_generation_results)}"

iid_index = list(iid_index)
ood_index = list(ood_index)

print(f"There are {len(iid_index)} iid generations and {len(ood_index)} ood generations")

# after added the task type, save the generation results
with open(text_generation_results_path, 'w') as f:
    json.dump(text_generation_results, f, indent=4, ensure_ascii=False)

import os
import torch

# for text_generations
for file in [
    text_generation_results_path,
]:
    results = json.load(open(file))
    
    assert 'task' in results[0], "Please add the task type to the generation results first"
    
    iid_results = [results[i] for i in iid_index]
    ood_results = [results[i] for i in ood_index]
    iid_file = file.replace('/full/', '/iid/')
    ood_file = file.replace('/full/', '/ood/')
    
    # make the folder if not exist
    os.makedirs(os.path.dirname(iid_file), exist_ok=True)
    os.makedirs(os.path.dirname(ood_file), exist_ok=True)
    
    assert 'task' in iid_results[0], 'Please add the task type to the generation results first'
    assert 'task' in ood_results[0], 'Please add the task type to the generation results first'
    
    json.dump(iid_results, open(iid_file, 'w'), indent=2)
    json.dump(ood_results, open(ood_file, 'w'), indent=2)
    
# for hidden_states
for file in [
    hidden_states_results_path,
]:
    results = torch.load(file)
    iid_results = results[iid_index]
    ood_results = results[ood_index]
    iid_file = file.replace('/full/', '/iid/')
    ood_file = file.replace('/full/', '/ood/')
    
    # make the folder if not exist
    os.makedirs(os.path.dirname(iid_file), exist_ok=True)
    os.makedirs(os.path.dirname(ood_file), exist_ok=True)
    
    torch.save(iid_results, iid_file)
    torch.save(ood_results, ood_file)




