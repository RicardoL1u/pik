import json
import argparse
import hashlib
import random
import glob
import pandas as pd
from collections import defaultdict
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

def generate_one_dpo_example(task, golden_rationale, bad_rationale):
    chosen_rationale = golden_rationale['rationale'][0]['rationale']
    reject_rationale = random.choice(list(bad_rationale['rationale'].values()))
    return {
        "id": golden_rationale["id"],
        "task": task,
        "instruction": task2instr[task],
        "input": golden_rationale["input"],
        "output":[chosen_rationale, reject_rationale],
        "is_correct": golden_rationale['rationale'][0]["is_correct"]
    }
    
    
def get_the_intersections_of_two_lists(list1, list2):
    return list(set(list1) & set(list2))



def process_task(task, use_correct_rationale):
    # get the golden rationale
    golden_rationale = json.load(open(f'data/bbh_cot/Qwen1.5-72B-chat/{task}_result.json'))
    # add id to the examples
    golden_rationale = [add_id_to_example(example) for example in golden_rationale]
    # sort the examples by id
    golden_rationale = sorted(golden_rationale, key=lambda x: x["id"])
    
    
    # get the bad rationale
    bad_rationale = json.load(open(f'data/bbh_dpo/{task}_rationale.json'))
    # add id to the examples
    bad_rationale = [add_id_to_example(example) for example in bad_rationale]
    # sort the examples by id
    bad_rationale = sorted(bad_rationale, key=lambda x: x["id"])
    
    # get the intersection of the two lists
    common_ids = get_the_intersections_of_two_lists([example["id"] for example in golden_rationale], [example["id"] for example in bad_rationale])
    print(f"Task {task} has {len(common_ids)} common ids")
    
    # filter the examples
    golden_rationale = [example for example in golden_rationale if example["id"] in common_ids]
    bad_rationale = [example for example in bad_rationale if example["id"] in common_ids]
    
    
    assert len(golden_rationale) == len(bad_rationale), f"Task {task} has different number of examples in golden {len(golden_rationale)} and bad {len(bad_rationale)} rationales"
    assert all(golden["id"] == bad["id"] for golden, bad in zip(golden_rationale, bad_rationale)), "Golden and bad rationales are not aligned"
    
    dpo_dataset = [generate_one_dpo_example(task, golden, bad) for golden, bad in zip(golden_rationale, bad_rationale) if not use_correct_rationale or golden["rationale"][0]["is_correct"]]
    
    return dpo_dataset
    
    

def main():
    args = parse_args()
    ood_task_list = json.load(open(args.ood_task_list))
    one_in_all_dpo_dataset = []
    for file in glob.glob('data/bbh_cot/Qwen1.5-72B-chat/*_result.json'):
        task = file.split('/')[-1].replace('_result.json', '')
        if task in ood_task_list:
            print(f"Skipping OOD task {task}")
            continue
        dpo_dataset = process_task(task, args.use_correct_rationale)
        with open(f'data/bbh_dpo_finetune_data/{task}_dpo.json', 'w') as f:
            json.dump(dpo_dataset, f, indent=4, ensure_ascii=False)
        print(f"Task {task} has {len(dpo_dataset)} examples")
        one_in_all_dpo_dataset.extend(dpo_dataset)
    
    random.shuffle(one_in_all_dpo_dataset)
    output_file = f'data/bbh_training_data/one_in_all_dpo_{args.use_correct_rationale}.json'
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
    
    task_analysis.to_csv(f'data/bbh_training_data/one_in_all_dpo_{args.use_correct_rationale}_analysis.csv')
    
    
    pass

if __name__ == "__main__":
    main()
    


