import json
import argparse
import logging
from collections import Counter
# from utils import evaluate_bbh, evaluate_answer_gsm8k, evaluate_answer_mcq_cmqa
import re
def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', required=True, help='dataset to use')
    args.add_argument('--result_file', required=True, help='path to result file')
    args.add_argument('--debug', action='store_true', help='Debug mode.')
    
    return args.parse_args()

def set_logging(args):
    logger = logging.getLogger(__name__)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    # logging all the args
    logging.info("===== Args =====")
    for arg in vars(args):
        logging.info("{}: {}".format(arg, getattr(args, arg)))
    return logger


def set_extract_function(dataset:str):
    if dataset == 'bbh':
        return extract_answer_for_bbh
    elif dataset == 'commonsense_qa':
        return extract_answer_for_commonsense_qa
    elif dataset == 'trivia_qa':
        return None
    else:
        raise NotImplementedError('Dataset not supported')
    
    
def extract_answer_for_bbh(model_answer:str):
    extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", model_answer)
    if extracted_answer:
        prediction = extracted_answer.group(1).strip()
    else:
        prediction = model_answer.strip()
    return prediction

def extract_answer_for_commonsense_qa(model_answer:str):
    # extract answer from "answer is Choice 3:" or "answer is choice 3" or "answer is choice3"
    model_answer = model_answer.lower()
    extracted_answer = re.search(r"[choice|option] ?(\d+)", model_answer)
    if extracted_answer:
        prediction = extracted_answer.group(1).strip()
    else:
        prediction = model_answer.strip()
    return prediction



if __name__ == '__main__':
    args = parse_arguments()
    set_logging(args)
    extract_function = set_extract_function(args.dataset)
    
    
    with open(args.result_file, 'r') as f:
        results = json.load(f)
    
    ### HACK: for trivia_qa, we use the evaluation as consistency
    if args.dataset == 'trivia_qa':
        for sample in results:
            sample['consistency'] = sample['evaluation']
        with open(args.result_file, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        exit()
        

    for sample in results:
        sample['extracted_answers'] = []
        # extract answer
        for model_answer in sample['model_answers']:
            sample['extracted_answers'].append(extract_function(model_answer))
        
        # count the answer frequency
        counter = Counter(sample['extracted_answers'])
        # save as dict
        sample['answer_count'] = dict(counter)
        # keep the most frequent answer
        sample['most_frequent_answer'] = (counter.most_common(1)[0][0], counter.most_common(1)[0][1])
        # calculate the consistency w.r.t most frequent answer
        sample['consistency'] = counter.most_common(1)[0][1] / len(sample['extracted_answers'])
        

    with open(args.result_file, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)