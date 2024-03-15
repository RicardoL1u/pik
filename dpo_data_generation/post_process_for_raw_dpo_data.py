import json
import glob
from tqdm import tqdm
import re
from collections import defaultdict, Counter

error_file_count = defaultdict(Counter)

# Find out all generated rationale file
rationale_files = glob.glob('data/bbh-new/*_result.json')




# Load the rationale file
for file in rationale_files:
    print("File: ", file)
    rationale_list = json.load(open(file))
    og_dataset = json.load(open(file.replace('_result.json', '.json')))
    assert len(rationale_list) == len(og_dataset), f"Length of rationale list {len(rationale_list)} and og_dataset {len(og_dataset)} are not same"
    for rationale, data in tqdm(zip(rationale_list, og_dataset), total=len(rationale_list), ncols=100):
        rationale = rationale.strip()
        if rationale.startswith('```json\n'):
            rationale = rationale.replace('```json\n', '')
        if rationale.endswith('\n```'):
            rationale = rationale.replace('\n```', '')
        try:
            data['rationale'] = json.loads(rationale)
        except Exception as e:
            error_file_count[file][type(e)] += 1
            continue
            print(f"Error {type(e)}: {e}")
            print("Error in file: ", file)
            print("Rationale: ", rationale)
            if isinstance(e,json.JSONDecodeError):
                # get column number and line number
                print(f"Line: {e.lineno}, Column: {e.colno}")
                # print context
                print(f"Context: {rationale.splitlines()[e.lineno-1][e.colno-10:e.colno+10]}")
            
            print("Data: ", data)
            print("=====================================")
            
            # use split to find the correct json
            # ```json\n{\n    \"meticulous\": \"To solve the given expression step by step, first perform the addition and subtraction within each set of parentheses separately:\n        (-9 - 4 - 7 + -9) = -29\n        (-4 - 9 - 5 + 5) = -13\n    Then, multiply the results together:\n        -29 * -13 = 377\n    Therefore, the result of ((-9 - 4 - 7 + -9) * (-4 - 9 - 5 + 5)) is 377.\",\n\n    \"succinct\": \"Multiply the values inside the parentheses first, then calculate the final result:\n        (-9 - 4 - 7 + -9) = -29\n        (-4 - 9 - 5 + 5) = -13\n        -29 * -13 = 377\",\n\n    \"profession\": \"When evaluating the expression, follow the order of operations (PEMDAS/BODMAS). First, simplify the expressions within the parentheses, then perform the multiplication:\n        (-9 - 4 - 7 + -9) = -29\n        (-4 - 9 - 5 + 5) = -13\n        -29 * -13 = 377\n    The result of the given expression is 377.\",\n\n    \"rookie-friendly\": \"First, add and subtract within each set of parentheses:\n        (-9 - 4 - 7 + -9) = -29\n        (-4 - 9 - 5 + 5) = -13\n    Then, multiply the results:\n        -29 * -13 = 377\n    Thus, the answer is 377.\"\n}\n```
            
            # delete possible ```json\n and \n```
            rationale = rationale.replace('```json\n', '').replace('\n```', '')
            # split by \"meticulous\": to get the correct json
            meticulous_rationale = rationale.split('\"meticulous\":')[1].split('\"succinct\":')[0]
            succinct_rationale = rationale.split('\"succinct\":')[1].split('\"profession\":')[0]
            profession_rationale = rationale.split('\"profession\":')[1].split('\"rookie-friendly\":')[0]
            rookie_friendly_rationale = rationale.split('\"rookie-friendly\":')[1]
            
            print("meticulous_rationale: ", meticulous_rationale)
            print("succinct_rationale: ", succinct_rationale)
            print("profession_rationale: ", profession_rationale)
            print("rookie_friendly_rationale: ", rookie_friendly_rationale)
            
            # "meticulous": "detailed and thorough",
            # "succinct": "brief and to the point",
            # "profession": "professional viewpoint",
            # "rookie-friendly": "accessible to beginners"
            # remove possible \"detailed and thorough\": or \"brief and to the point\": or \"professional viewpoint\": or \"accessible to beginners\": in the beginning from rationales in both lower and upper case
            meticulous_rationale = re.sub(r'^\"detailed and thorough\":|\"detailed and thorough\":', '', meticulous_rationale, flags=re.IGNORECASE)
            succinct_rationale = re.sub(r'^\"brief and to the point\":|\"brief and to the point\":', '', succinct_rationale, flags=re.IGNORECASE)
            profession_rationale = re.sub(r'^\"professional viewpoint\":|\"professional viewpoint\":', '', profession_rationale, flags=re.IGNORECASE)
            rookie_friendly_rationale = re.sub(r'^\"accessible to beginners\":|\"accessible to beginners\":', '', rookie_friendly_rationale, flags=re.IGNORECASE)
            
            # strip the rationale
            meticulous_rationale = meticulous_rationale.strip()
            succinct_rationale = succinct_rationale.strip()
            profession_rationale = profession_rationale.strip()
            rookie_friendly_rationale = rookie_friendly_rationale.strip()
               
            # remove all sorts of special characters exclude numbers and alphabets from the beginning and end of the string
            meticulous_rationale = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', meticulous_rationale)
            succinct_rationale = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', succinct_rationale)
            profession_rationale = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', profession_rationale)
            rookie_friendly_rationale = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', rookie_friendly_rationale)
            
            
            # add the json to the data
            data['rationale'] = {
                "meticulous": meticulous_rationale,
                "succinct": succinct_rationale,
                "profession": profession_rationale,
                "rookie-friendly": rookie_friendly_rationale
            }
            
            print("After postprocess, Data with rationale: ", json.dumps(data, indent=4))
            
            
    dpo_dataset = [data for data in og_dataset if 'rationale' in data.keys()]
    print(f"Total data: {len(og_dataset)}, Data with rationale: {len(dpo_dataset)}")
    assert all('rationale' in data for data in dpo_dataset), "Rationale is missing in some data"
    
    # save the file
    dpo_dataset_path = file.replace('_result.json', '_rationale.json').replace('bbh-new', 'bbh_dpo')           
    with open(dpo_dataset_path, 'w') as f:
        json.dump(dpo_dataset, f, indent=4, ensure_ascii=False)
        
    # open the dumped file and check if the data is saved correctly
    # dpo_dataset = json.load(open(dpo_dataset_path))
    # assert all('rationale' in data for data in dpo_dataset), "Rationale is missing in some data"
    
# print error file count
print("Error file count: ")
for file, error_cnt in error_file_count.items():
    print(f"File: {file}")
    for error, cnt in error_cnt.items():
        print(f"Error: {error}, Count: {cnt}")
    print("=====================================")