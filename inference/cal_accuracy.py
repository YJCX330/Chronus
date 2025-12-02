from tqdm import tqdm
import json
import os


##videomme
results_path="./results/videomme_long.json"
with open('./data/test/videomme_long_wo_audio.json', 'r') as file:
    data_ref = json.load(file)

with open(results_path, 'r') as file:
    data_gen = json.load(file)
assert len(data_ref)==len(data_gen)
qa_num=len(data_gen)
right_num=0 
for i in range(len(data_gen)):
    assert data_ref[i]["id"]==data_gen[i]["id"]
    if data_ref[i]["conversations"][1]["value"].lower()==data_gen[i]["output"][0].lower():
        right_num=right_num+1
print(right_num/qa_num)
print(right_num)

