# Chronus
ChronusOmni: Improving Time Awareness of Omni Large Language Models

## TODO
- [x] Release model code
- [x] Release checkpoint
- [x] Release dataset
- [ ] Release dataset pipeline code

## ChronusAV Dataset
Download annotation files: [ChronusAV](https://huggingface.co/datasets/mxxxxxxxxxxxxxxxxx/ChronusAV)

Videos are selected from [Panda-70M](https://github.com/snap-research/Panda-70M)

Videos_url: https://www.youtube.com/watch?v={video_id}

## ChronusOmni

### Installation
```bash
conda create -n chronusomni python=3.10 -y
conda activate chronusomni
pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 --index-url https://download.pytorch.org/whl/cu118 
pip install -r requirements.txt
# https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.0/flash_attn-2.6.0+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.6.0+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Evaluation
Download [chronusomni checkpoint](https://huggingface.co/mxxxxxxxxxxxxxxxxx/ChronusOmni)
```bash
cd Chronus
export PYTHONPATH=./
python3 inference/eval.py --json_file ./data/test/v2a_openqa.json --results_path ./results/chronus_v2a_open.json
```
Evaluation data format:
```bash
[
  {
    "id": id,
    "video": video_path,
    "audio": audio_path,
    "question": question
  },
  ...
] 
```
### Training
Download [Ola checkpoint](https://huggingface.co/THUdyh/Ola-7b) to initialize

SFT: 
```bash
cd Chronus
export PYTHONPATH=./
bash ./scripts/finetune_ola.sh
```

SFT training data format:
```bash
[
    {
        "id": id,
        "video": video_path,
        "audio": audio_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<speech><image>\n{user_query}"
            },
            {
                "from": "gpt",
                "value": ground_truth_answer
            }
        ],
    },
    ...
]
```

GRPO:
```bash
cd Chronus
export PYTHONPATH=./
bash ./scripts/train_grpo.sh
```
GRPO training data format:
```bash
[
    {
        "id": id,
        "data_type": data_type, # data_type is one of: "t2a", "a2t", "t2v", "v2t", "a2v", "v2a"
        "video": video_path,
        "audio": audio_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<speech><image>\n{user_query}"
            },
            {
                "from": "gpt",
                "value": ground_truth_answer
            }
        ],
    },
    ...
]
```
Our training json file can be found at [training json file](https://huggingface.co/datasets/mxxxxxxxxxxxxxxxxx/ChronusOmni_training_json_file)

## Citation
If you find it useful for your research and applications, please cite our paper using this BibTeX:
```bibtex
@article{chronusomni,
title={ChronusOmni: Improving Time Awareness of Omni Large Language Models},
author={Chen, Yijing and Wu, Yihan and Guan, Kaisi and Ren, Yuchen and Wang, Yuyue and Song, Ruihua and Ru, Liyun},
journal={arXiv preprint arXiv:2512.09841},
year={2025}
}
```

## Acknowledgement
Our code is conducted on [Ola](https://github.com/Ola-Omni/Ola)
