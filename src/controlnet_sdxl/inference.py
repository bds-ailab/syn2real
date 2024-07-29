from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import argparse


def main(controlnet_path, cond_image_path, prompt, out_path):
    """Generates realistic image from conditioning image (segmentation maps + Canny edges)

    Args:
        controlnet_path (str): path to ControlNet weights path
        cond_image_path (str): path to conditioning image
        prompt (str): text prompt
        out_path (str): output image saving path
    """
    # import base model from hub
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

    # Construct controlnet model
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=torch.float16
    )
    # Construct Pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch.float16
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed or when using Torch 2.0.
    # pipe.enable_xformers_memory_efficient_attention()
    # memory optimization.
    pipe.enable_model_cpu_offload()

    # Load test image
    control_image = load_image(cond_image_path).resize((1024, 512))

    # generate image
    generator = torch.manual_seed(0)
    image = pipe(
        prompt, num_inference_steps=20, generator=generator, image=control_image
    ).images[0]

    image.save(out_path)


if __name__ == "__main__":

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
        "--prompt",
        type=str,
        default="a real image of cars driving down the road",
        required=True,
        help="text prompt",
    )

    parser.add_argument(
        "--controlnet_path",
        type=str,
        default=None,
        required=True,
        help="text prompt",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        default="output.png",
        required=True,
        help="output image saving path ",
    )

    args = parser.parse_args()
    main(args.controlnet_path, args.cond_image_path, args.prompt, args.out_path)
