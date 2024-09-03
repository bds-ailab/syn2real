#!/bin/bash

export OUTPUT_DIR="/out/controlnet_sdxl_active_round2_distortion/"
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export CONTROLNET_DIR="/out/controlnet_sdxl_active_round1/"

python -c "from accelerate.utils import write_basic_config; write_basic_config()"

accelerate launch /sdxl/train_controlnet_sdxl.py --controlnet_model_name_or_path=$CONTROLNET_DIR --pretrained_model_name_or_path=$MODEL_DIR  --output_dir=$OUTPUT_DIR  --train_data_dir=/data/cityscapes --mixed_precision="fp16"  --resolution=1024 --report_to="wandb"  --learning_rate=4e-5  --max_train_steps=20000  --train_batch_size=2  --gradient_accumulation_steps=10  --seed=42 --use_8bit_adam --set_grads_to_none --dataloader_num_workers=25 --proportion_empty_prompts=0.2 --canny_edges