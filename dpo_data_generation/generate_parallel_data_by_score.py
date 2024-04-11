import json
import argparse
import hashlib
import random
import glob
import pandas as pd
from collections import defaultdict
from typing import List
random.seed(42)
task2instr = json.load(open("data/bbh_dpo/template/task2instr.json"))

def parse_args():
    parser = argparse.ArgumentParser(description="Generate parallel data for DPO")
    parser.add_argument("--use_correct_rationale", action="store_true", help="Use correct rationale")
    parser.add_argument("--ood_task_list", type=str, help="json file containing the list of OOD tasks")
    return parser.parse_args()

def add_id_to_example(example:dict):
    meta_data = {
        "input": example["input"],
        "target": example["target"],
    }
    id = hashlib.sha256(json.dumps(meta_data).encode()).hexdigest()
    return {"id": id, **example}

def generate_dpo_examples(task, origin_example):
    
    to_chosen_rationales = [r for r in origin_example['rationale'] if r['is_correct']]
    full_rationales = origin_example['rationale']
    
    dpo_examples = []
    for chosen_rationale in to_chosen_rationales:
        chosen_rationale_text = chosen_rationale['rationale']
        chosen_score = chosen_rationale['consistency']
        
        for rationale in full_rationales:
            reject_rationale_text = rationale['rationale']
            reject_score = rationale['consistency']
            if reject_score > chosen_score or reject_rationale_text == chosen_rationale_text: 
                continue
            dpo_examples.append({
                "id": origin_example["id"] + f"_{len(dpo_examples)}",
                "task": task,
                "instruction": task2instr[task],
                "input": origin_example["input"],
                "output":[chosen_rationale_text, reject_rationale_text],
                "is_correct": [chosen_rationale['is_correct'], rationale['is_correct']],
                "score": [chosen_score, reject_score],
            })
    
    return dpo_examples
    
    
    
def get_the_intersections_of_two_lists(list1, list2):
    return list(set(list1) & set(list2))



def process_task(task, use_correct_rationale):
    
    
    examples = json.load(open(f'temp_data/bbh/cot/bbh-new/gpt-3.5-turbo/cot-prompts-3-shot/rating/{task}_rationale.json'))
    
    # add id to the examples
    examples = [add_id_to_example(example) for example in examples]
    # sort the examples by id
    examples = sorted(examples, key=lambda x: x["id"])
    
    # filter out the examples that are not correct
    if use_correct_rationale:
        examples = [ex for ex in examples if any([r['is_correct'] for r in ex['rationale']])]
    
    
    dpo_dataset = sum([generate_dpo_examples(task, rationale) for rationale in examples ], [])
    
    return dpo_dataset
    
    

def main():
    args = parse_args()
    ood_task_list = json.load(open(args.ood_task_list))
    one_in_all_dpo_dataset = []
    for task in json.load(open('dpo_data_generation/task.json')):
        if task in ood_task_list:
            print(f"Skipping OOD task {task}")
            continue
        dpo_dataset = process_task(task, args.use_correct_rationale)
        with open(f'temp_data/bbh/training/{task}_dpo.json', 'w') as f:
            json.dump(dpo_dataset, f, indent=4, ensure_ascii=False)
        print(f"Task {task} has {len(dpo_dataset)} examples")
        one_in_all_dpo_dataset.extend(dpo_dataset)
    
    random.shuffle(one_in_all_dpo_dataset)
    output_file = f'temp_data/bbh/training/one_in_all_dpo_{args.use_correct_rationale}.json'
    with open(output_file, 'w') as f:
        json.dump(one_in_all_dpo_dataset, f, indent=4, ensure_ascii=False)
    
    # export analysis in csv with dataframe, there are 5 columns:
    # task, number of examples, number of correct examples, number of incorrect examples, percentage of correct examples
    task_analysis = defaultdict(dict)
    for example in one_in_all_dpo_dataset:
        task = example["task"]
        if task not in task_analysis:
            task_analysis[task] = {
                "total": 0,
                "correct": 0,
                "incorrect": 0
            }
        task_analysis[task]["total"] += 1
        if example["is_correct"]:
            task_analysis[task]["correct"] += 1
        else:
            task_analysis[task]["incorrect"] += 1
    task_analysis = pd.DataFrame(task_analysis).T
    task_analysis["percentage"] = task_analysis["correct"] / task_analysis["total"]
    
    # sort by the dictionary order of the task
    task_analysis = task_analysis.sort_index()
    
    task_analysis.to_csv(f'temp_data/bbh/training/one_in_all_dpo_{args.use_correct_rationale}_analysis.csv')
    
    
    pass

if __name__ == "__main__":
    main()
    


