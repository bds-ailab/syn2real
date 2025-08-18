#!/bin/bash

# export NCCL_P2P_DISABLE=1
accelerate config default

accelerate launch --main_process_port 29504 /gan/train_cyclegan.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="/out/cyclegan_20000_steps_lowres_lambdaidt2p5/" \
    --dataset_folder "/data/cityscapes_cyclegan" \
    --train_img_prep "pers_resize" --val_img_prep "no_resize" \
    --learning_rate="1e-5" --max_train_steps=20000 \
    --train_batch_size=2 --gradient_accumulation_steps=4 \
    --report_to "wandb" --tracker_project_name "unpaired_cyclegan_20000_steps_lowres_lambdaidt2p5" \
    --validation_steps 40 --lambda_gan 0.5 --lambda_idt 2 --lambda_cycle 1 --dataloader_num_workers 30
    