
import glob
import json
import openai_proxy
import re
from tqdm import tqdm
training_files = glob.glob('data/bbh-new/*.json')

training_files = [f for f in training_files if not f.endswith('_prompt.json') and not f.endswith('_result.json')]

for file in training_files:
    # Task:
    task = file.split('/')[-1].split('.')[0]
    print("Task: ", task)
    
    print("File: ", file)
    dataset = json.load(open(file))
    # load the cot prompt from data/bbh/bbh/cot-prompts
    cot_prompt = open(f'data/bbh/cot-prompts/{task}.txt').read()
    
    prompt_list = []
    result_list = []
    for data in tqdm(dataset, desc=f"Processing {file}",ncols=100):
        input_prompt = cot_prompt + '\n\nQ: ' + data['input'] + '\nA: Let\'s think step by step.'
        # print(input_prompt)
        rationales = openai_proxy.chat_completion_use_cache(input_prompt, temperature=1, n=3)
        # data['rationale'] = rationales
        
        # extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", model_answer)
        answers = [re.search(r"[t|T]he answer is (.*?)\.", rationale) for rationale in rationales]
        is_correct_list = []
        for answer in answers:
            if answer:
                is_correct_list.append(answer.group(1).strip().split()[0].lower() == data['target'].lower())
            else:
                is_correct_list.append(False)
        data['rationale'] = [
            {
                'rationale': rationale,
                'answer': answer.group(1).strip() if answer else None,
                'is_correct': is_correct
            }
            for rationale, answer, is_correct in zip(rationales, answers, is_correct_list)
        ]
        
        prompt_list.append(input_prompt)
    with open(f'data/bbh_cot/{task}_prompt.json', 'w') as f:
        json.dump(prompt_list, f, indent=4, ensure_ascii=False)
    with open(f'data/bbh_cot/{task}_result.json', 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    


