
import re
import string
import torch
import json
# Prompt engineering
PREAMBLE = ''
XSHOT_TEMPLATE = 'Question: {question}\nAnswer: {answer}'
POSTAMBLE = 'Question: {question}\nAnswer: '
EAXMPLES = json.load(open('data/trivia_qa/example_qa_pairs.json', 'r'))
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

 
def build_few_shot(example_num=0):
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
        question, answer = EAXMPLES[i]
        example_list.append(XSHOT_TEMPLATE.format(question=question, answer=answer))
        # prompt += XSHOT_TEMPLATE.format(question=question, answer=answer)
    prompt = '\n'.join(example_list)
    return prompt

def prompt_eng(question, example_num=0):
    '''
    Returns an x-shot prompt for the given question.
    If `n` is higher than 0, `dataset` must be provided.
    '''
    return PREAMBLE + build_few_shot(example_num) + '\n' + POSTAMBLE.format(question=question)

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


def evaluate_answer(model_answer, dataset_answer, exact_match=False):
    '''
    Returns 1 (correct) if `dataset_answer` is (a substring of) `model_answer`
    Returns 0 (incorrect) otherwise
    '''
    # use alias in dataset_answer
    all_possible_answer = dataset_answer['normalized_aliases'] + [dataset_answer['normalized_value']]
    if exact_match:
        return any([model_answer == normalize_answer(ans) for ans in all_possible_answer])
    return any([normalize_answer(model_answer) in normalize_answer(ans) for ans in all_possible_answer])
