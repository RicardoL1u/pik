import os
import json
from tqdm import tqdm

DATA_PATH = "../data/20240304"
ORI_DATA_PATH = "ori"
MODEL_DATA_PATH = "model"

for file in os.listdir(os.path.join(DATA_PATH, ORI_DATA_PATH)):
    print(file)
    with open(os.path.join(DATA_PATH, ORI_DATA_PATH, file), "r") as f:
        data = json.load(f)
    with open(os.path.join(DATA_PATH, MODEL_DATA_PATH, file + "l"), "w") as f:
        for d in tqdm(data):
            temp_json = {}
            temp_json["query"] = d["input"]
            temp_json["response"] = d["output"]
            f.write(json.dumps(temp_json) + "\n")
