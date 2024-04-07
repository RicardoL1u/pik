import json
import logging
import argparse
import os
import hashlib 
import random
random.seed(42)
# set logging format to print the time and line number of the code
logging.basicConfig(format='%(asctime)s LINE %(lineno)d: %(message)s', level=logging.INFO)


def process_task(task:str, input_dir:str, output_dir:str):
    """
    Process a task
    """
    cot_prompt = open(os.path.join('data/bbh/cot-prompts-0-shot', f'{task}.txt')).read()
    origin_dataset = json.load(open(os.path.join(input_dir, f'{task}.json')))
    processed_dataset = []
    for example in origin_dataset:
        if example['rationale'][0]['is_correct'] == False:
            continue
        input_prompt = cot_prompt + '\n\nQ: ' + example['input'] + '\nA: Let\'s think step by step.'
        response = example['rationale'][0]['rationale']
        
        processed_dataset.append({
            'id': hashlib.md5((input_prompt + response).encode()).hexdigest(),
            'task': task,
            'input': input_prompt,
            'output': response,
        })
    with open(os.path.join(output_dir, f'{task}.json'), 'w') as f:
        json.dump(processed_dataset, f, indent=4, ensure_ascii=False)

    return processed_dataset
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SFT data')
    parser.add_argument('--input_dir', type=str, default='data/sft-prompts', help='Directory for SFT prompts')
    parser.add_argument('--output_dir', type=str, default='data/sft-new', help='Target directory for SFT data')
    
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    tasks: list = json.load(open('dpo_data_generation/task.json'))
    
    total_dataset = sum([process_task(task, input_dir, output_dir) for task in tasks], [])

    logging.info(f"Total number of examples: {len(total_dataset)}")
    
    random.shuffle(total_dataset)
    with open(os.path.join(output_dir, 'all.json'), 'w') as f:
        json.dump(total_dataset, f, indent=4, ensure_ascii=False)