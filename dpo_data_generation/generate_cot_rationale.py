
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
from collections import defaultdict


def extract_options_content(string):
    # Regex pattern
    pattern = r'\([A-Z]\)\s+(.*)'

    # Extract options and dates
    matches = re.findall(pattern, string)

    # Print options and dates
    # for i, match in enumerate(matches):
    #     print(f"Option ({chr(65 + i)}): {match}")
    
    return {
        f"({chr(65 + i)})": match.strip()
        for i, match in enumerate(matches)
    }

def extract_answer_from_rationale(rationale:str)->str:
    """
    Extract the answer from the rationale
    rationale: str - the rationale
    """
    # remove possible \n and . in the end of the rationale
    rationale = re.sub(r'[\n.]+$', '', rationale)
    
    answer_split_dot = rationale.split('.')
    answer_split_new_line = rationale.split('\n\n')
    
    # use the one with the least number of words in the last sentence
    if len(answer_split_dot[-1].split()) < len(answer_split_new_line[-1].split()) and \
        len(answer_split_dot[-1].split()) > 1: # special check for the last sentence with only 1 word
        answer = answer_split_dot[-1]
    else:
        answer = answer_split_new_line[-1]
    return answer

FULL_ALPHABET_OPTIONS = {'(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)'}

def check_response(response:str, input:str, target:str, task:str)->bool:
    """
    Check if the response is correct
    response: str - the response from the model
    target: str - the target answer
    task: str - the task name
    """
    ## extract the answer from the rationale
    answer = extract_answer_from_rationale(response)
    is_correct = None
    if target == 'valid':
        return 'valid' in answer and 'invalid' not in answer
    elif target == 'invalid':
        return 'invalid' in answer
    elif task == 'web_of_lies':
        if target == 'No':
            is_correct = ('not' in answer and 'tell' in answer and 'truth' in answer) or ('False' in answer)
        elif target == 'Yes':
            is_correct = ('tell' in answer and 'truth' in answer and 'not' not in answer) or ('True' in answer)
    elif task == 'sports_understanding':
        not_pasuible = ('not' in answer and 'pausible' in answer) or ('not' not in answer and 'implausible' in answer) or ('unlikely' in answer) or ('unusual' in answer) or ('not' in answer and 'accurate' in answer) or ("incorrect" in answer)
        # target would be 'no' and 'yes'
        is_correct = (target == 'no' and not_pasuible) or (target == 'yes' and not not_pasuible)
    # elif task == 'dyck_languages':  
    elif target in FULL_ALPHABET_OPTIONS:
        options_content = extract_options_content(input)
        all_option_content = set(options_content.values())
        target_content = options_content[target]
        is_correct = (target in answer and all([option not in answer for option in FULL_ALPHABET_OPTIONS - {target}])) or \
            (target_content in answer and all([option not in answer for option in all_option_content - {target_content}]))
    else:
        is_correct = target in answer
    
    assert is_correct is not None, f"Task {task} is not implemented"
    return is_correct
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate COT rationales')
    parser.add_argument('--model_checkpoint', type=str, default='/workspace/MODELS/Qwen1.5-72B-chat', help='Model checkpoint')
    parser.add_argument('--prompt_dir', type=str, default='data/bbh/cot-prompts', help='Directory for COT prompts')
    parser.add_argument('--target_bbh_dir', type=str, default='data/bbh-new', help='Target directory for BBH data')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint
    model_name = Path(model_checkpoint).name
    prompt_folder_name = Path(args.prompt_dir).name
    target_bbh_dir = args.target_bbh_dir
    # output dir would be data/bbh-new/{MODEL_NAME}/{prompt_folder}_{temperature}/{task}.json
    output_dir = os.path.join(target_bbh_dir, model_name, f'{prompt_folder_name}_{args.temperature}')
    temperature = args.temperature

    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Model name: {model_name}")
    print(f"Target BBH directory: {target_bbh_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Prompt directory: {args.prompt_dir}")


    # Load the model
    if model_checkpoint == 'gpt-3.5-turbo':
        llm = None
    else:
        llm = Model(
            model_checkpoint=args.model_checkpoint,
            generation_options= {
                "max_new_tokens": 512,
                "temperature": temperature,
            },
            is_low_memory = False,
            is_chat_model = 'chat' in model_name.lower()
        )

    # llm = None



    training_files = glob.glob(os.path.join(target_bbh_dir, '*.json'))

    training_files = [f for f in training_files if not f.endswith('_prompt.json') and not f.endswith('_result.json')]


    task_data = defaultdict(dict)

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
            temp_dataset = json.load(open(os.path.join(output_dir, f'{task}.json')))        
            if "example" in temp_dataset:
                temp_dataset = temp_dataset["example"]
            if len(temp_dataset) == len(dataset):
                print(f"Do not generate for {task}")
                do_generate = False
                dataset = temp_dataset
            else:
                print(f"Re-generate for {task}")
                del temp_dataset
        
        print(f"Do generate: {do_generate}")
        
        if args.debug:
            dataset = dataset[10:20]
        
        print(f"Number of examples: {len(dataset)}")
        for data in tqdm(dataset, desc=f"Processing {file}",ncols=100):
            input_prompt = cot_prompt + '\n\nQ: ' + data['input'] + '\nA: Let\'s think step by step.'
            # print(input_prompt)
            if model_checkpoint == 'gpt-3.5-turbo':
                rationales = openai_proxy.chat_completion_use_cache(input_prompt, temperature=1, n=3)
            else:
                if do_generate:
                    rationales = llm.get_text_generation(input_prompt)
                else:
                    rationales = [r['rationale'] for r in data['rationale']]

            
            # extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", model_answer)
            # answers = [rationale.split('.')[-1] for rationale in rationales]
            data['rationale'] = []
            for rationale in rationales:
                is_correct = check_response(rationale, data['input'], data['target'], task)
                data['rationale'].append({
                    'rationale': rationale,
                    'answer': extract_answer_from_rationale(rationale),
                    'is_correct': is_correct
                })
            
        task_data[task]['accuracy'] = sum([r['rationale'][0]['is_correct'] for r in dataset]) / len(dataset)
        task_data[task]['num_examples'] = len(dataset)
        
        with open(os.path.join(output_dir, f'{task}.json'), 'w') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)

    
    # convert task2accuracy to pandas dataframe [task, accuracy, num_examples]
    df = pd.DataFrame(task_data).T
    # df['task'] = df.index
    # convert to percentage then round to 2 decimal places
    df['accuracy'] = (df['accuracy'] * 100).round(2)

    # sort by task
    df = df.sort_index()

    print("The accuracy of the model on each task:")
    print(df)

    # save to csv
    df.to_csv(os.path.join(output_dir, 'accuracy.csv'), index=True)

