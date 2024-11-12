#!/bin/bash
export IMAGE_PATH="/data/cityscapes_syn/images/"
accelerate config default

python /gan/inference.py --model_path "/out/cyclegan_40000_steps_lowres/checkpoints/model_20001.pkl" \
    --input_image $IMAGE_PATH \
    --prompt "a real picture of a street from dashcam point of view" --direction "a2b" \
    --output_dir "/out/cyclegan_inference_lambda1/" --image_prep "pers_resize"