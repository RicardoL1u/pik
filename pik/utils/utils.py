
import re
import string
import torch
import json
from datasets import load_dataset as load_dataset_hf
from typing import List
# Prompt engineering
PREAMBLE = ''
import wandb
import logging

def wandb_log(logging_level, use_wandb=False, log_msg=None, 
              score_dict=None, step=None):
    '''
    Logs a string to wandb.
    '''
    if log_msg is not None:
        logging.log(logging_level, log_msg)
    if score_dict is not None:
        logging.log(logging_level, score_dict)
        if use_wandb:
            wandb.log(score_dict, step=step)

 
def build_few_shot(examples:List[dict], template:str, example_num=0):
    '''
    Designed to following the prompting format seen section A.7 of the paper
    'Language Models (Mostly) Know What They Know'
    https://arxiv.org/pdf/2207.05221.pdf
    '''
    if example_num == 0:
        return ''
    prompt = ''
    example_list = []
    for i in range(example_num):
        question = examples[i].get('question', '')
        rationale = examples[i].get('rationale', '')
        answer = examples[i].get('answer', '')
        example_list.append(template.format(
            question=question, rationale=rationale, answer=answer))
        # prompt += XSHOT_TEMPLATE.format(question=question, answer=answer)
    prompt = '\n'.join(example_list)
    return prompt

def prompt_eng(question:str, examples:List[dict], template:str, example_num=0):
    '''
    Returns an x-shot prompt for the given question.
    If `n` is higher than 0, `dataset` must be provided.
    '''
    
    # template would be like 'Question: {question}\nAnswer: {answer}'
    # or 'Question: {question}\nRationale: {rationale}\nAnswer: {answer}'
    # we need to find the second space in the template
    position_of_second_space = template.find(' ', template.find(' ') + 1)
    # only keep the "Question: {question}\nAnswer|Rationale: "
    postamble = template[:position_of_second_space] + ' '
    return PREAMBLE + build_few_shot(examples, template, example_num) + '\n' + postamble.format(question=question)

def normalize_answer(s):
    '''
    Lower text and remove punctuation, articles and extra whitespace.
    Taken from official triviaqa eval script.
    https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
    '''
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def handle_punc(text):
        exclude = set(string.punctuation + ''.join([u'‘', u'’', u'´', u'`']))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    def lower(text):
        return text.lower()
    def replace_underscore(text):
        return text.replace('_', ' ')
    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def evaluate_answer_trivia_qa(model_answer, dataset_answer, exact_match=False):
    '''
    Returns 1 (correct) if `dataset_answer` is (a substring of) `model_answer`
    Returns 0 (incorrect) otherwise
    '''
    # use alias in dataset_answer
    all_possible_answer = dataset_answer['normalized_aliases'] + [dataset_answer['normalized_value']]
    min_answer_len = min([len(ans) for ans in all_possible_answer])
    if exact_match:
        return any([model_answer == normalize_answer(ans) for ans in all_possible_answer])
    normalized_model_answer = normalize_answer(model_answer)
    return any([ans in normalized_model_answer for ans in all_possible_answer])

def evaluate_answer_gsm8k(model_answer:str, dataset_answer:str, exact_match=True):
    '''
    Returns 1 (correct) if `dataset_answer` is (a substring of) `model_answer`
    Returns 0 (incorrect) otherwise
    '''
    # replace numbers like `x,xxx` with `xxxx`
    model_answer = re.sub(r"(\d),(\d)", r"\1\2", model_answer)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", model_answer)
    
    # if numbers is not empty
    if numbers:
        model_answer = numbers[-1]
    if exact_match:
        return model_answer.strip() == dataset_answer.strip()
    else:
        raise NotImplementedError('For GSM8K, exact_match must be True')

def load_dataset(dataset_name: str)->(List[dict], List[dict], callable):
    '''
    Loads the dataset from the given directory.
    '''
    if dataset_name == 'trivia_qa_wiki':
        dataset = load_dataset_hf('data/trivia_qa_wiki', split='validation')
        evaluate_answer = evaluate_answer_trivia_qa
    elif dataset_name == 'gsm8k':
        ori_dataset = [json.loads(line) for line in open('data/gsm8k/train.jsonl', 'r')]
        ori_dataset.extend([json.loads(line) for line in open('data/gsm8k/test.jsonl', 'r')])
        dataset = []
        for data in ori_dataset:
            dataset.append({
                'question': data['question'],
                'rationale': data['answer'],
                # answer is like "600>>5,600 more calories.\n#### 5,600"}
                'answer': data['answer'].split('\n#### ')[-1].replace(',','').strip()
            })
        
        evaluate_answer = evaluate_answer_gsm8k
    else:
        raise NotImplementedError(f'Unknown dataset: {dataset_name}')
    return dataset, evaluate_answer

def load_example(example_path:str)->List[dict]:
    '''
    Loads the example from the given directory.
    '''
    examples = json.load(open(example_path, 'r'))
    return examples

def load_template(template_type:str):
    '''
    Loads the template from the given directory.
    '''
    if template_type.lower().strip() == 'icl':
        return 'Question: {question}\nAnswer: {answer}'
    elif template_type.lower().strip() == 'cot':
        return 'Question: {question}\nRationale: {rationale}\nAnswer: {answer}'
    else:
        raise NotImplementedError(f'Unknown template: {template_type}')