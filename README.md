# Chronus
ChronusOmni: Improving Time Awareness of Omni Large Language Models

## TODO
- [ ] Release model code
- [ ] Release checkpoint
- [ ] Release dataset
- [ ] Release dataset pipeline code

## ChronusAV Dataset

## Installation
conda create -n chronusomni python=3.10 -y
conda activate chronusomni
pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 --index-url https://download.pytorch.org/whl/cu118 
pip install -r requirements.txt
# https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.0/flash_attn-2.6.0+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.6.0+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

## Evaluation
Download [chronusomni checkpoint](https://huggingface.co/mxxxxxxxxxxxxxxxxx/ChronusOmni)
cd Chronus
export PYTHONPATH=./
python3 inference/eval.py --json_file ./data/test/v2a_openqa.json --results_path ./results/chronus_v2a_open.json

## Training
Download [Ola checkpoint](https://huggingface.co/THUdyh/Ola-7b) to initialize
SFT: 
cd Chronus
export PYTHONPATH=./
bash ./scripts/finetune_ola.sh
GRPO:
cd Chronus
export PYTHONPATH=./
bash ./scripts/train_grpo.sh

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
