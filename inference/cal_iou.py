from tqdm import tqdm
import json
import os
import re


def calculate_iou(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start) 
    
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start 
    
    iou = intersection / union if union > 0 else 0 
    return iou


##chronus
results_path="./results/chronus_v2t_open.json"
with open('./data/test/v2t_openqa.json', 'r') as file:
    data_ref = json.load(file)

with open(results_path, 'r') as file:
    data_gen = json.load(file)
print(len(data_gen),len(data_ref))
iou_all=[] 
for i in range(len(data_gen)):
    assert data_gen[i]["id"]==data_ref[i]["id"]
    matches_gen = re.findall(r'(\d+\.?\d*)', data_gen[i]["output"])
    matches_gen = [float(matches_gen[i]) for i in range(len(matches_gen)) if (i==0 or i==len(matches_gen)-1)]
    matches_ref = re.findall(r'\{(\d+\.?\d*)\}', data_ref[i]["conversations"][1]["value"])
    matches_ref = [float(match) for match in matches_ref]
    # print(matches_gen,matches_ref)
    try:
        iou=calculate_iou(matches_gen,matches_ref)
    except Exception:
        iou_all.append(0)
        continue
    iou_all.append(iou)
print(len(iou_all))
print(sum(1 for iou in iou_all if iou >= 0.5)/len(iou_all))
print(sum(1 for iou in iou_all if iou >= 0.7)/len(iou_all))
# print(sum(iou_all) / len(iou_all))

