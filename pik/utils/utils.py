
import re
import string
import glob
import torch
import json
from datasets import load_dataset as load_dataset_hf
from typing import List, Tuple
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

def prompt_eng_for_freeqa(sample:dict, examples:List[dict], template:str, example_num=0):
    '''
    Returns an x-shot prompt for the given question.
    '''
    
    question = sample['question']
    # template would be like 'Question: {question}\nAnswer: {answer}'
    # or 'Question: {question}\nRationale: {rationale}\nAnswer: {answer}'
    # we need to find the second space in the template
    position_of_second_space = template.find(' ', template.find(' ') + 1)
    # only keep the "Question: {question}\nAnswer|Rationale: "
    postamble = template[:position_of_second_space] + ' '
    
    # HACK: if the template is {question} namely only for BBH dataset
    # we keep the postamble as {question}
    if template == '{question}':
        postamble = '{question}'
    return PREAMBLE + build_few_shot(examples, template, example_num) + '\n' + postamble.format(question=question)

def prompt_eng_for_mcq(sample:dict, examples:List[dict], template:str, example_num=0):
    
    # TODO: NOW, the examples is not used
    # TODO: NOW, the choice number must be 5
    question = sample['question']
    choices = sample['choices']['text']
    return PREAMBLE + build_few_shot(examples, template, 0) + '\n' + template.format(**{
        'test_question': question,
        'option_1': choices[0],
        'option_2': choices[1],
        'option_3': choices[2],
        'option_4': choices[3],
        'option_5': choices[4]
    })


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

def evaluate_answer_mcq_cmqa(model_answer:str, dataset_answer:dict, exact_match=False):
    # the dataset_answer is like 
    # {
    #   "choices": {
    #       "label": ["A", "B", "C", "D", "E"],
    #       "text": ["bloody mess", "pleasure", "being imprisoned", "feeling of guilt", "cake"]
    #   },
    #   "answerKey": "A"
    # }
    
    # First we need to convert answerKey to the index of the choice
    answer_idx = dataset_answer['choices']['label'].index(dataset_answer['answerKey'])
    # idx starts from 0, while the answer starts from 1
    answer_text = dataset_answer['choices']['text'][answer_idx]
    answer_idx += 1
    other_choices = list(set(range(1, 6)) - set([answer_idx]))
    
    if exact_match:
        raise NotImplementedError('For MCQ_CMQA, exact_match must be False')
    else:
        # Then we need to check if the answer_idx is in the model_answer
        # Meanwhile, other choices should not be in the model_answer
        if str(answer_idx) in model_answer and not any([str(idx) in model_answer for idx in other_choices]):
            return True
        else:
            # check if the answer_text is in the model_answer
            if answer_text.lower() in model_answer.lower():
                return True
            else:
                return False
        

def evaluate_bbh(model_answer:str, dataset_answer:str, exact_match=False):
    extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", model_answer)
    if extracted_answer:
        prediction = extracted_answer.group(1).strip()
    else:
        prediction = model_answer.strip()
    
    if exact_match:
        raise NotImplementedError('For BBH, exact_match must be False')
    else:
        return prediction.lower() in dataset_answer.lower()


def load_dataset(dataset_name: str) -> Tuple[List[dict], List[dict], callable]:
    '''
    Loads the dataset from the given directory.
    '''
    if dataset_name == 'trivia_qa_wiki':
        dataset = load_dataset_hf('data/trivia_qa_wiki/rc.wikipedia.nocontext', split='train')
        evaluate_answer = evaluate_answer_trivia_qa
    if dataset_name == 'commonsense_qa':
        dataset = load_dataset_hf('data/commonsense_qa', split='train')
        evaluate_answer = evaluate_answer_mcq_cmqa
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
    elif dataset_name == 'bbh':
        task_files = glob.glob('data/bbh/bbh/*.json')
        # cot_prompt_files = glob.glob('data/bbh/cot_prompts/*.txt')
        dataset = []
        for task_file in task_files:
            cot_prompt = open('data/bbh/cot-prompts-1-shot/' + task_file.split('/')[-1].split('.')[0] + '.txt', 'r').read().strip()
            # since BBH is a set of tasks, the cot_prompt varies for each task
            # we load the cot_prompt for each task at here, rather than in the prompt_eng
            
            examples = json.load(open(task_file, 'r'))['examples']
            for example in examples:
                dataset.append({
                    'task': task_file.split('/')[-1].split('.')[0],
                    'question': cot_prompt + "\n\nQ: " + example['input'] + "\nA: Let's think step by step.\n",
                    'answer': example['target']
                })
        evaluate_answer = evaluate_bbh

        
    else:
        raise NotImplementedError(f'Unknown dataset: {dataset_name}')
    return dataset, evaluate_answer

def load_example(example_path:str)->List[dict]:
    '''
    Loads the example from the given directory.
    '''
    # for bbh, we don't need to load the example
    if 'bbh' in example_path:
        return ''
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
    # multiple choice question for commonsense_qa
    elif template_type.lower().strip() == 'mcq_cmqa':
        return open('data/commonsense_qa/data/template.txt', 'r').read()
    elif template_type.lower().strip() == 'bbh':
        return '{question}'
    else:
        raise NotImplementedError(f'Unknown template: {template_type}')