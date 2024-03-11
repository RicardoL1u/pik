
import json
import glob 
from convert_to_bbh import transform_to_bbh

# find all json files in the directory
origin_files = glob.glob('data/bbh-full/*.json')
bbh_files = glob.glob('data/bbh/bbh/*.json')


tasks = [file.split('/')[-1].split('.')[0] for file in origin_files]
print(f"There are {len(tasks)} tasks to be processed")

assert len(tasks) == len(bbh_files), "The number of tasks and the number of bbh files are not equal"


import random
random.seed(42)

for task, origin_file, bbh_file in zip(tasks, origin_files, bbh_files):
    print(f"Processing {task} with {origin_file} and {bbh_file}")
    origin_data = json.load(open(origin_file))
    bbh_data = json.load(open(bbh_file))
    # find out data in origin_data that is not in bbh_data
    new_data = []
    for example in origin_data['examples']:
        if all([example['input'] not in e['input'] for e in bbh_data['examples']]):
            new_data.append(example)
    # random sample 100 examples
    new_data = random.sample(new_data, min(100, len(new_data)))
    print(f"Found {len(new_data)} new examples from {len(origin_data['examples'])} examples")
    # convert to bbh format
    # new_data = [transform_to_bbh(task, example) for example in new_data]
    new_data_cp = []
    for example in new_data:
        # print(example)
        try:
            new_data_cp.append(transform_to_bbh(task, example))
        except Exception as e:
            print(e)
            print("Task: ", task)
            print("Example: ", example)
        
            print(f"Failed to convert {json.dumps(example, indent=4)}" in {task})
            exit()

    new_data = new_data_cp
    print(f"Converted to {len(new_data)} examples")
    with open(f'data/bbh-new/{task}.json', 'w') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)




