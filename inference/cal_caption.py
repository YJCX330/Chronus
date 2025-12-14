import json
from collections import defaultdict
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def caption_metrics(data_gen, data_ref):
    pds = defaultdict(list)
    gts = defaultdict(list)
    for i, gen1 in enumerate(data_gen):
        video_id = gen1['id']
        pds[i].append({"video_id": video_id, "caption":gen1['output']})
    for j,ref1 in enumerate(data_ref):
        video_id = ref1['id']
        gts[j].append({"video_id": video_id, "caption":ref1["conversations"][1]["value"]})


    tokenizer = PTBTokenizer() 
    pds = tokenizer.tokenize(pds)
    gts = tokenizer.tokenize(gts)
    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
               (Meteor(),"METEOR"),
               (Rouge(), "ROUGE_L"),
               (Cider(), "CIDEr")]
    eval_dict = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, pds)
        if scorer.method() == "Bleu":
            eval_dict["BLEU4"] = score[3]
        else:
            eval_dict[scorer.method()] = score
    return eval_dict


if __name__ == "__main__":
    data_gen=json.load(open("./results/chronus_v2a_open.json","r"))
    data_ref=json.load(open("./data/test/v2a_openqa.json","r"))
    assert len(data_gen)==len(data_ref)
    eval_dict = caption_metrics(data_gen, data_ref)
    for k, v in eval_dict.items():
        print("%s: %.4f"%(k, v)) 