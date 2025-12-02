export LOWRES_RESIZE=384x32
export VIDEO_RESIZE="0x32"
export HIGHRES_BASE="0x32"
export MAXRES=1536
export MINRES=0
export VIDEO_MAXRES=448
export VIDEO_MINRES=288
export PAD2STRIDE=1
export FORCE_NO_DOWNSAMPLE=1
export LOAD_VISION_EARLY=1

export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
EXP_NAME="train_grpo1"

CHECKPOINT='./log/checkpoints/sft_70k/checkpoint-4375'
DATA="/data/chronus_GRPO.json"

torchrun --nproc_per_node=4 --nnodes="1" --node_rank="0" --master_addr="127.0.0.1" --master_port="12351" \
    chronus/train/train_grpo.py \
    --deepspeed ./scripts/ds_config_zero3_offload_grpo.json \
    --run_name $EXP_NAME \
    --model_name_or_path $CHECKPOINT \
    --pretrain_speech_projector $CHECKPOINT/speech_projector.bin \
    --vision_tower /123 \
    --mm_projector_type chronus_mlp \
    --speech_projector_type "linear" \
    --mm_vision_select_layer -1 \
    --mm_use_im_patch_token False \
    --tune_speech_adapter False \
    --version qwen_1_5 \
    --data_path $DATA  \
    --bf16 True \
    --output_dir ./log/checkpoints/$EXP_NAME \
    --fix_speech_encoder True \
    --speech_encoder "./checkpoints/large-v3.pt" \
    --music_encoder "./checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" \
    --speech_encoder_type "dual" \
    --speech_encoder_hidden_size 2048 \
    --speech_encoder_ds_rate 10 \
    --max_prompt_length 4096 \
    --max_completion_length 1024 \
    --learning_rate 1e-6 \
    --beta 0.04 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --save_steps 500 \
    --save_only_model false \
    --save_total_limit 4 \
    --num_generations 4 \
    --dataset_name "123" #\
    # --report_to none 
