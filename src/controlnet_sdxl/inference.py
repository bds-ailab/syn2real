from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import argparse
import os
import json
from os import path
import cv2


def main(pipe, cond_image_paths, prompts, seed):
    """Generates realistic image from conditioning image (segmentation maps + Canny edges)

    Args:
        controlnet_path (str): path to ControlNet weights path
        cond_image_path (str): path to conditioning image
        prompt (str): text prompt
        out_path (str): output image saving path
    """

    # Load test image
    control_images = [
        load_image(path).convert("RGB").resize((1024, 512)) for path in cond_image_paths
    ]

    # generate image
    generator = torch.manual_seed(seed)
    images = pipe(
        prompts,
        num_inference_steps=20,
        generator=generator,
        image=control_images,
    ).images

    return images


def sharpen(image_path, out_path):
    """Sharpen generated images using simple filter

    Args:
        image_path (str): path of image to sharpen
        out_path (str): path to save sharpened image

    """

    # Load the image
    image = cv2.imread(image_path)

    # Create the sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Sharpen the image
    sharpened_image = cv2.filter2D(image, -1, kernel)

    # Adjust Brightness and Contrast
    corrected_image = correct(sharpened_image, 280, 150)

    # Save the image
    cv2.imwrite(out_path, corrected_image)


def correct(img, brightness=255, contrast=127):
    """Corrects brightness and contrast of generated images

    Args:
        img (ndarray): Input image
        brightness (int, optional): Brightness level. Defaults to 255.
        contrast (int, optional): _Contrast level. Defaults to 127.

    Returns:
        ndarray: Corrected image
    """

    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:

        if brightness > 0:

            shadow = brightness

            max = 255

        else:

            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha, img, 0, ga_mma)

    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma)

    return cal


if __name__ == "__main__":

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(
        description="Inference example of trained ControlNet XL"
    )

    parser.add_argument(
        "--cond_image_path",
        type=str,
        default="/data/cond1.png",
        required=True,
        help="path to conditioning image",
    )

    parser.add_argument(
        "--controlnet_path",
        type=str,
        default=None,
        required=True,
        help="ControlNet weights path",
    )

    parser.add_argument(
        "--unet_path",
        type=str,
        default=None,
        required=True,
        help="Unet weights path",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        default="output.png",
        help="output image saving path ",
    )

    args = parser.parse_args()

    # import base model from hub
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

    # Load Controlnet weights
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_path, torch_dtype=torch.float16
    )
    # Load Unet weights
    unet = UNet2DConditionModel.from_pretrained(
        args.unet_path, subfolder="unet", torch_dtype=torch.float16
    )
    # Construct Pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_path,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # memory optimization.
    pipe.enable_model_cpu_offload()

    # Read conditioning images
    cond_images = [
        Image.open(args.cond_image_path).convert("RGB"),
    ]

    # Different colors for generation variability
    colors = ["white", "green", "red", "blue", "black", "grey", "yellow"]

    # Generate multiple examples
    for i in range(20):
        # Choose a color at random
        c = np.random.choice(colors)
        # Detailed prompt
        prompts = [
            f"a real picture of a city street with a {c} car and bus driving down it in a sunny day. not artifacted."
        ]
        # Inference
        aug_images = main(pipe, cond_images, prompts, seed=i)
        # Save results
        out_img_path = path.join(args.out_path, f"out{i}.png")
        aug_images[0].save(out_img_path)
        # Sharpen generated images (optional)
        sharpen(out_img_path, out_img_path)

    # Example command :
    # Checkpoints used for paper v1 : python inference.py --cond_image_path='sam_inference/mask.png' --controlnet_path='/out/controlnet_sdxl_active_round2_distortion/checkpoint-20000/controlnet/' --unet_path='/out/controlnet_sdxl_unlocked_attention/unet/' --out_path='sam_inference/'
    # Last Checkpoints of controlnet without segmentation for real images (just canny) : python inference.py --cond_image_path='sam_inference/mask.png' --controlnet_path='/out/controlnet_withouseg4real/round2_controlnet/checkpoint-5000/controlnet/' --unet_path='/out/controlnet_withouseg4real/round1_unet/checkpoint-5000/' --out_path='sam_inference/'
    # Last Checkpoints of controlnet sam segmentation for all images (+ canny) : python inference.py --cond_image_path='sam_inference/mask_sam.png' --controlnet_path='/out/controlnet_sam/round2_controlnet/checkpoint-5000/controlnet/' --unet_path='/out/controlnet_sam/round2_unet/checkpoint-5000/' --out_path='sam_inference/'
    # Last Checkpoints of controlnet with human segmentation for synthetic and SAM for real images (+canny) : python inference.py --cond_image_path='sam_inference/mask.png' --controlnet_path='/out/controlnet_mix/round1_controlnet/checkpoint-20000/controlnet/' --unet_path='/out/controlnet_withouseg4real/round1_unet/checkpoint-5000/' --out_path='sam_inference/'
    # Last Checkpoints of controlnet on bdd dataset (generates artifacts) : python inference.py --cond_image_path='sam_inference/mask.png' --controlnet_path='/out/controlnet_bdd/round1_controlnet/checkpoint-20000/controlnet/' --unet_path='/out/controlnet_withouseg4real/round1_unet/checkpoint-5000/' --out_path='sam_inference/'
