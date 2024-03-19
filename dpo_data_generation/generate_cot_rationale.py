
import glob
import json
import pik.utils.openai_proxy as openai_proxy
import re
from tqdm import tqdm
from pik.models.model import Model
import os


# Load the model
llm = Model(
    model_checkpoint='MODELS/Qwen/Qwen1.5-72B-Chat-GPTQ-Int4',
    generation_options= {
        "max_new_tokens": 512,
        "temperature": 1.0,
    },
    is_low_memory = False,
    is_chat_model = True
)


training_files = glob.glob('data/bbh-new/*.json')

training_files = [f for f in training_files if not f.endswith('_prompt.json') and not f.endswith('_result.json')]


task2accuracy = {}

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
    
    # check if the result file exists
    do_generate = True
    if os.path.exists(f'data/bbh_cot/Qwen1.5-72B-chat/{task}_result.json'):
        print(f"Skipping {task}")
        do_generate = False
        dataset = json.load(open(f'data/bbh_cot/Qwen1.5-72B-chat/{task}_result.json'))
    
    
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
        
    with open(f'data/bbh_cot/Qwen1.5-72B-chat/{task}_result.json', 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

with open(f'data/bbh_cot/Qwen1.5-72B-chat_accuracy.json', 'w') as f:
    json.dump(task2accuracy, f, indent=4, ensure_ascii=False)


