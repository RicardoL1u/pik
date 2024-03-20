from transformers import AutoTokenizer
from datasets import Dataset
import json

def load_dpo_dataset(
    sanity_check: bool = False,
    tokenizer: AutoTokenizer = None,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    


    with open("./one_in_all_dpo.json", "r") as f:
        data = json.load(f)

    final_data_list = []

    for d in data:
        temp_json = {}
        prompt = f"{d['instruction']}\n\nQ: {d['input']}\nA: Let's think step by step."
        chat_msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        temp_json["prompt"] =  tokenizer.apply_chat_template(chat_msg, tokenize=False, add_generation_prompt=True)
        output = d["output"]
        temp_json["chosen"] = output[0]
        temp_json["rejected"] = output[1]
        final_data_list.append(temp_json)

    final_data_dict = {}
    final_data_dict["prompt"] = [d["prompt"].strip() for d in final_data_list]
    final_data_dict["chosen"] = [d["chosen"].strip() for d in final_data_list]
    final_data_dict["rejected"] = [d["rejected"].strip() for d in final_data_list]

    dataset = Dataset.from_dict(final_data_dict)

    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
    
    return dataset

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/data2/MODELS/Qwen1.5-0.5B-chat")
    dataset = load_dpo_dataset(sanity_check=False, tokenizer=tokenizer).train_test_split(test_size=0.001, seed=42)
    # split the dataset
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print("Dataset: ", dataset)
    