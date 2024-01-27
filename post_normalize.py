import pandas as pd
from pik.utils.utils import normalize_answer, evaluate_answer

# 7b, 13b ,30b, 65b
for model in ['llama-7b-hf', 'llama-13b-hf', 'llama-30b-hf', 'llama-65b-hf']:
    text_generations = pd.read_csv(f'data/{model}/text_generations.csv')
    qa = pd.read_csv(f'data/{model}/qa_pairs.csv')

    # drop the row with null answer in qa
    print('Before dropna:', qa.shape)
    qa = qa.dropna(subset=['answer'])
    print('After dropna:', qa.shape)
    
    
    question_list = qa['question'].tolist()
    # need to repeat each question 30 times
    repeat_question_list = []
    for question in question_list:
        for i in range(30):
            repeat_question_list.append(question)

    # same to answer
    answer_list = qa['answer'].tolist()
    repeat_answer_list = []
    for answer in answer_list:
        for i in range(30):
            repeat_answer_list.append(answer)

    # add two column called 'question' and 'answer' into text_generations
    text_generations['question'] = repeat_question_list
    text_generations['answer'] = repeat_answer_list

    # set the null in model_answer to ''
    text_generations['model_answer'] = text_generations['model_answer'].fillna('')

    # strip model_answer
    text_generations['model_answer'] = text_generations['model_answer'].apply(lambda x: x.strip())
    
    # drop the row with null in answer
    print('Before dropna:', text_generations.shape)
    text_generations = text_generations.dropna(subset=['answer'])
    
    # drop the row with null in question
    text_generations = text_generations.dropna(subset=['question'])
    print('After dropna:', text_generations.shape)
    
    # reevaluate based on the answer in text_generations['answer']
    text_generations['evaluation'] = text_generations.apply(lambda x: evaluate_answer(x['model_answer'], x['answer']), axis=1)

    # print the mean of evaluation
    print('Mean evaluation score:', text_generations.evaluation.mean())

    # save the new text_generations
    text_generations.to_csv(f'data/{model}/text_generations.csv', index=False)