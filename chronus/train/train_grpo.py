from collections import defaultdict
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
import traceback
from typing import Optional
from jiwer import wer
import nltk
# nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score
import jieba
from datasets import load_dataset, Dataset, DatasetDict
import string
from chronus_trainer_grpo import ChronusGRPOUniTrainer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import numpy as np

import json


@dataclass
class ModelArguments(ModelConfig):
    # LLM Arguments
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    pretrained_safetensor_path: Optional[str] = field(default=None)
    resume_from: Optional[str] = field(default=None)
    version: Optional[str] = field(default="qwen_1_5")
    s2s: bool = field(default=False)
    speech_audio: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
    tune_speech_adapter: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    speech_encoder: Optional[str] = field(default=None)
    music_encoder: Optional[str] = field(default=None)
    fix_speech_encoder: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_speech_projector: Optional[str] = field(default=None)
    speech_projector_type: Optional[str] = field(default='none')
    speech_encoder_type: Optional[str] = field(default='none')
    speech_encoder_config: Optional[str] = field(default='')
    speech_encoder_ds_rate: Optional[int] = field(default=10)
    speech_encoder_hidden_size: Optional[int] = field(default=1280)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["v2t_iou","v2t_format","a2t_iou","a2t_format","v2a_meteor","t2a_meteor","a2v_meteor","t2v_meteor"],
        metadata={"help": "List of reward functions. "},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "data json file path"},
    )



def t_format_reward(completions, **kwargs):
    pattern = r"second\{[0-9]+(\.[0-9]+)?\}-second\{[0-9]+(\.[0-9]+)?\}"
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completions]
    return [1.0 if match else 0.0 for match in matches]

def parse_timestamp_output(output_string):
    matches_gen = re.findall(r'\{(\d+\.?\d*)\}', output_string)
    start_time, end_time = [float(matches_gen[i]) for i in range(len(matches_gen))]
    return start_time, end_time

def iou_timestamp_reward(completions, solution, **kwargs): # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    print(completions, solution)
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content in completions: 
        reward = 0.0
        gt_start, gt_end = parse_timestamp_output(solution)
        try:
            start_time, end_time = 0, 0
            parsed_times = parse_timestamp_output(content)
            s, e = gt_start, gt_end
            if parsed_times:
                start_time, end_time = parsed_times
                from_number = start_time
                to_number = end_time

                intersection = max(0, min(to_number, e) - max(from_number, s))
                union = max(to_number, e) - min(from_number, s)
                if union > 0:
                    iou = intersection / union   # 0.1 0.3
                    reward = iou             
        except Exception:
            reward = 0.0
        print('gt second:', gt_start, gt_end)
        print('pred second:', start_time, end_time)  
        print(f"------------- {current_time} IoU reward: {reward} -------------\n")
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"Content: {content}\n")
                f.write(f"pred second: {str(start_time)}, {str(end_time)}\n")
                f.write(f"gt second: {str(gt_start)}, {str(gt_end)}\n")
                f.write(f"------------- {current_time} IoU reward: {reward} -------------\n") # Modified log message
    return rewards

def wer_reward(completions, solution, **kwargs):
    rewards = []
    punctuation=string.punctuation
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content in completions: 
        reward = 0.0
        try:
            wer_score=1
            hypothesis=''.join([char for char in list(content.lower()) if char not in punctuation])
            groundtruth=''.join([char for char in list(solution.lower()) if char not in punctuation])
            wer_score=wer(groundtruth,hypothesis)
            if wer_score<1:
                reward=1-wer_score
        except Exception:
            reward = 0.0
        print('wer:', wer_score,hypothesis,groundtruth)  
        print(f"------------- {current_time} Wer reward: {reward} -------------\n")
        rewards.append(reward)
    return rewards

def a_meteor_reward(completions, solution, **kwargs):
    rewards = []
    punctuation=string.punctuation
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content in completions: 
        reward = 0.0
        try:
            meteor=0.0
            groundtruth = jieba.lcut(solution)
            groundtruth=[token for token in groundtruth if token!=' ']
            hypothesis = jieba.lcut(content)
            hypothesis=[token for token in hypothesis if token!=' ']
            meteor = meteor_score([groundtruth], hypothesis)
            reward=meteor
        except Exception:
            reward = 0.0
        print('a_meteor:', meteor, content,"-------",solution)  
        print(f"------------- {current_time} Meteor reward: {reward} -------------\n")
        rewards.append(reward)
    return rewards

def v_meteor_reward(completions, solution, **kwargs):
    rewards = []
    punctuation=string.punctuation
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content in completions: 
        reward = 0.0
        try:
            meteor=0.0
            groundtruth = jieba.lcut(solution)
            groundtruth=[token for token in groundtruth if token!=' ']
            hypothesis = jieba.lcut(content)
            hypothesis=[token for token in hypothesis if token!=' ']
            meteor = meteor_score([groundtruth], hypothesis)
            reward=meteor
        except Exception:
            reward = 0.0
        print('v_meteor:', meteor, content,"-------",solution)  
        print(f"------------- {current_time} Meteor reward: {reward} -------------\n")
        rewards.append(reward)
    return rewards


reward_funcs_registry = {
    "v2t_iou":iou_timestamp_reward,
    "v2t_format":t_format_reward,
    "a2t_iou":iou_timestamp_reward,
    "a2t_format":t_format_reward,
    "v2a_meteor":a_meteor_reward,
    "t2a_meteor":a_meteor_reward,
    "a2v_meteor":v_meteor_reward,
    "t2v_meteor":v_meteor_reward,
}


def create_dataset_from_jsonl_simple(jsonl_path):
    base_dataset = Dataset.from_json(jsonl_path)
    return DatasetDict({
        "train": base_dataset
    })


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = {}
    for func in script_args.reward_funcs:
        task_type = func.split('_')[0]
        if task_type in reward_funcs:
            reward_funcs[task_type].append(reward_funcs_registry[func])
        else:
            reward_funcs[task_type] = [reward_funcs_registry[func]]

    train_dataset = create_dataset_from_jsonl_simple(script_args.data_path)

    def make_conversation(example):
        if "prompt" in example.keys():
            return {"prompt":example["prompt"]}
        else:
            return {"prompt":None}
    train_dataset = train_dataset.map(make_conversation)

    trainer_cls = ChronusGRPOUniTrainer
    # import bitsandbytes as bnb
    # optimizer_cls = bnb.optim.AdamW8bit

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model_id=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset['train'],
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels    )

    checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]

    if checkpoints:
    # Train and push the model to the Hub
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelArguments))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args )
