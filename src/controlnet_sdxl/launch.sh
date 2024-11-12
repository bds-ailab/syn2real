#!/bin/bash

export OUTPUT_DIR="/out/controlnet_bdd/round1_controlnet/"
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export CONTROLNET_DIR="None"
export UNET_DIR="None"

python -c "from accelerate.utils import write_basic_config; write_basic_config()"

# Uncomment if wanting to train only controlnet
accelerate launch /sdxl/train_controlnet_sdxl.py --controlnet_model_name_or_path=$CONTROLNET_DIR --pretrained_model_name_or_path=$MODEL_DIR  --output_dir=$OUTPUT_DIR  --train_data_dir=/data/gta_bdd --mixed_precision="fp16"  --resolution=1024 --report_to="wandb"  --learning_rate=4e-5  --max_train_steps=20000  --train_batch_size=2  --gradient_accumulation_steps=10  --seed=42 --use_8bit_adam --set_grads_to_none --dataloader_num_workers=25 --proportion_empty_prompts=0.2 --canny_edges --segmentation
# Uncomment if wanting to unlock unet layers of SD 
# accelerate launch /sdxl/train_sdxl_unlocked.py --controlnet_model_name_or_path=$CONTROLNET_DIR --unet_model_name_or_path=$UNET_DIR --pretrained_model_name_or_path=$MODEL_DIR  --output_dir=$OUTPUT_DIR  --train_data_dir=/data/cityscapes_new --mixed_precision="fp16"  --resolution=1024 --report_to="wandb"  --learning_rate=1e-5  --max_train_steps=5000  --train_batch_size=2  --gradient_accumulation_steps=10  --seed=42 --use_8bit_adam --set_grads_to_none --dataloader_num_workers=25 --proportion_empty_prompts=0.2 --canny_edges --segmentation