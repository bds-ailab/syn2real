from transformers import T5EncoderModel
from diffusers import DiffusionPipeline
import gc
import torch
from PIL import Image
import torchvision.transforms as transforms
import json
import os
import yaml
import argparse


def flush():
    """Clean GPU memory after finishing with some pipelines"""
    # Collect garbage (deleted objects)
    gc.collect()

    # Empty torch memory
    # Note: the pipelines will still be stored in the CPU cache but not in the GPU's
    # if you want to re-use them, they will be loaded from the CPU to the GPU
    torch.cuda.empty_cache()


def encode_prompts(prompts):
    """Extract the embeddings of each prompt in the passed list. We use this method to load the text embedder
    once and free the memory for the diffusion pipelines

    Args:
        prompts (list): list of input prompts for the images to be upscaled

    Returns:
        (list): list of the extracted embeddings
    """

    # Load the text encoder to extract the prompts embeddings
    text_encoder = T5EncoderModel.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0",
        subfolder="text_encoder",
        device_map="auto",
        load_in_8bit=True,
        variant="8bit",
    )
    # Load the pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0",
        text_encoder=text_encoder,  # pass the previously instantiated 8bit text encoder
        unet=None,
        device_map="balanced",
    )

    # Extract the embeddings of each prompt
    embeddings = []
    for text in prompts:
        prompt_embeds, negative_embeds = pipe.encode_prompt(text)
        embeddings.append((prompt_embeds, negative_embeds))

    # Delete the text encoder
    del text_encoder
    # Delete the pipeline
    del pipe

    # Flush the garbage
    flush()
    return embeddings


def transform_img(img_path, size=100):
    """Load, resize the image and transform it to tensors for the diffusion model

    Args:
        img_path (str): image absolute path
        size (int, optional): shape to resize. Defaults to 100.

    Returns:
        (torch.Tensor): transformed image tensor
    """

    # Load the image from the path
    original_image = Image.open(img_path)
    shape = original_image.size

    # Crop the image to a square image
    # For the moment huggingface pipeline for DF IF does not have
    # a feature to change the aspect ratio
    if shape[0] != shape[1]:
        min_lenght = min(list(shape))
        original_image = original_image.crop((0, 0, min_lenght, min_lenght))

    # Resize the image to lower resolution
    # Note: with this resize we lose some information especially of the background details But
    # since our model generates artifacts in the background, it is beneficial to reconstruct them better
    image = original_image.resize((size, size))

    # Define a transform to convert PIL
    # image to a Torch tensor
    transform = transforms.Compose([transforms.PILToTensor()])

    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)

    # Shift the pixels values to the interval [-1, 1]
    img_tensor = (img_tensor[None, :, :, :] / 255) * 2 - 1

    return img_tensor


def process(images_list, prompts_list, size=100, output_dir="/out/upscaling/"):
    """upscale the input images

    Args:
        images_list (list): list of absolute paths to images
        prompts_list (list): list of prompts to guide the model
        size (int, optional): shape of the resize image. Defaults to 100.
        output_dir (str, optional): path to the output folder. Defaults to "/out/upscaling/".
    """

    # Prompts embeddings extraction
    embeddings = encode_prompts(prompts_list)

    # Load the pipeline for the first stage upscaler
    pipe1 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0",
        text_encoder=None,  # no use of text encoder => memory savings!
        variant="fp16",
        torch_dtype=torch.float16,
        device_map="balanced",
    )

    # Load the pipeline for the second stage upscaler
    pipe2 = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        variant="fp16",
        torch_dtype=torch.float16,
        device_map="balanced",
    )

    # Seed
    generator = torch.Generator().manual_seed(2)

    # Iterate on the images list
    for i in range(len(images_list)):

        # load and prepare the image
        img_tensor = transform_img(images_list[i], size=size)

        # First upscaling to 256*256
        image = pipe1(
            image=img_tensor,
            prompt_embeds=embeddings[i][0],
            negative_prompt_embeds=embeddings[i][1],
            output_type="pt",
            generator=generator,
        ).images

        # Second upscaling to 1024*1024
        pil_image = pipe2(
            prompts_list[i], generator=generator, image=(image + 1) / 2
        ).images
        print(i)

        # Save output image
        pil_image[0].save(f"{output_dir}image_{i}.png")

    # Don't forget to flush kids! :)
    del pipe1
    flush()
    del pipe2
    flush()


if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run controlnet finetuning with specified parameters"
    )

    # Add config file path (yml file)
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yml",
        help="Path to configuration parameters (yaml file)",
    )

    args = parser.parse_args()

    # Read yaml configuration file
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    # Path to original dataset to get the BLIP captions
    data_path = config["data_path"]
    with open(data_path, "r") as f:
        data = json.load(f)

    # Collect the prompts
    prompts_list = [
        item["caption"] for item in data[config["start_idx"] : config["end_idx"]]
    ]

    # Paths to the generated images of our model with canny control version
    folder_path = config["images_path"]
    images_list = [
        folder_path + file for file in os.listdir(folder_path) if "out1" in file
    ]

    # Output folder
    output_path = config["output_path"]

    # Upscale !
    process(images_list, prompts_list, size=100, output_dir=output_path)
