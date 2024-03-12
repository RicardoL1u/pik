
import glob
import json
import openai_proxy
from tqdm import tqdm
training_files = glob.glob('data/bbh-new/*.json')

training_files = [f for f in training_files if not f.endswith('_prompt.json') and not f.endswith('_result.json')]


template = open('template.txt').read()


print(template)


for file in training_files:
    print("File: ", file)
    dataset = json.load(open(file))
    prompt_list = []
    result_list = []
    for data in tqdm(dataset, desc=f"Processing {file}",ncols=100):
        prompt = template.format(input=data['input'], target=data['target'])
        result = openai_proxy.chat_completion_use_cache(prompt)
        result_list.append(result)
        prompt_list.append(prompt)
    with open(file.replace('.json','_prompt.json'), 'w') as f:
        json.dump(prompt_list, f, indent=4, ensure_ascii=False)
    with open(file.replace('.json','_result.json'), 'w') as f:
        json.dump(result_list, f, indent=4, ensure_ascii=False)
    


