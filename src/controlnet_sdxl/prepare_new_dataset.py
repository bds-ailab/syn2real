from PIL import Image
import numpy as np
import cv2
import json
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
from os import path
import json


def parse_args():

    parser = argparse.ArgumentParser(
        description="Transform new (real or synthetic) dataset with pretrained models"
    )

    parser.add_argument(
        "--in_json_file",
        type=str,
        required=True,
        help="path to dataset json file ",
    )

    parser.add_argument(
        "--out_json_file",
        type=str,
        required=True,
        help="path to save transformed dataset json file",
    )

    parser.add_argument(
        "--out_dataset_folder",
        type=str,
        required=True,
        help="path to save transformed dataset",
    )

    parser.add_argument(
        "--unet_path",
        type=str,
        required=True,
        help="Unet pretrained path",
    )

    parser.add_argument(
        "--controlnet_path",
        type=str,
        required=True,
        help="Controlnet pretrained path",
    )

    args = parser.parse_args()
    return args


def read_json_data(json_file):
    """Read json and jsonl files data

    Args:
        json_file (str): path to json file

    Returns:
        list: loaded data
    """

    # Read the whole file if it's json
    if json_file.endswith("json"):
        with open(json_file, "r") as f:
            data = json.load(f)

    # Read line by line if it's json lines
    elif json_file.endswith("jsonl"):
        data = []
        with open(json_file, "r") as f:
            for line in f:
                data.append(json.loads(line))
    return data


def write_json_data(json_file, data):
    """Write data in json files

    Args:
        json_file (str): path to output json file
        data (list): data to be saved
    """

    # Save the whole data in file if it's json
    if json_file.endswith("json"):
        with open(json_file, "w") as outfile:
            json.dump(data, outfile)

    # Write line by line if it's json lines
    elif json_file.endswith("jsonl"):
        with open(json_file, "w") as outfile:
            for entry in data:
                json.dump(entry, outfile)
                outfile.write("\n")


def construct_conditioning_image(image, conditioning_image):
    """Construct conditioning image by superposing segmentation maps and Canny Edges

    Args:
        image (str): path to original image
        conditioning_image (str): path to segmentation map

    Returns:
        (pil_image, pil_image): segmentation map without and with Canny edges
    """

    # Load images
    image = Image.open(image).convert("RGB")
    conditioning_image = Image.open(conditioning_image).convert("RGB")

    # Transform to numpy arrays
    image_array = np.array(image)
    conditioning_image_array = np.array(conditioning_image)
    syn_bool = 1

    # Different thresholds for real in syn images because of the noise present in syn images
    if not syn_bool:
        low_threshold = 50
        high_threshold = 120

    # Thresholds were selected experimentaly by looking for values that generates the same output
    # for both real and synthetic images
    else:
        low_threshold = 250
        high_threshold = 350

    # Transforming the target image to canny edges
    canny_image = cv2.Canny(image_array, low_threshold, high_threshold)

    # Drawing thicker and more visible line with dilate
    kernel = np.ones((2, 2), np.uint8)
    canny_image = cv2.dilate(canny_image, kernel, iterations=1)
    canny_image = canny_image[:, :, None]

    # Formatting and normalizing the canny edges image to the same shape of source images
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2) / 255

    # Combining segmentation map with canny edges
    conditioning_image_array[canny_image == 1] = 255

    # Returning conditioning image in pil format
    trans_conditioning_image = Image.fromarray(conditioning_image_array).resize(
        (1024, 512)
    )

    return conditioning_image, trans_conditioning_image


def load_pipeline(args):
    """load pipeline from pretrained modules (controlnet, unet)

    Args:
        args (Parser.args): arguments with the paths to pretrained modules

    Returns:
        StableDiffusionXLControlNetPipeline: Pipeline for inference
    """
    # import base model from hub
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

    # Construct controlnet model
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_path, torch_dtype=torch.float16
    )
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

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # memory optimization.
    pipe.enable_model_cpu_offload()

    return pipe


def transform_img(pipe, batch):
    """transforms batch images using pipeline

    Args:
        pipe (StableDiffusionXLControlNetPipeline): Inference Pipeline
        batch (dict): Dictionnary with batch of images and prompts

    Returns:
        list: list of generated images (augmented)
    """
    # Extracting conditionning images and prompts
    conditioning_images, prompts = batch["im"], batch["text"]
    # generate image
    generator = torch.manual_seed(0)
    aug_images = pipe(
        prompts,
        num_inference_steps=20,
        generator=generator,
        image=conditioning_images,
    ).images
    return aug_images


def save_batch(args, out_data, batch, starting_idx):
    """Save a batch of generated images in pre-existing dataset folder

    Args:
        args (Parser.args): arguments with dataset infos
        out_data (list): list of json data about the existing dataset
        batch (dict): dictionnary with batch of images and prompts
        starting_idx (int): starting index is the number of images that already exist in the dataset folder

    Returns:
        list: list of json data with the additional files,
        int: incremented starting index
    """
    for i in range(len(batch["text"])):
        # Path to save generated image
        img_path = path.join(
            args.out_dataset_folder, f"images/image{starting_idx+i}.png"
        )
        # Path to save conditioning image
        cond_img_path = path.join(
            args.out_dataset_folder,
            f"conditioning_images/cond_image{starting_idx+i}.png",
        )
        # Save images
        batch["im"][i].save(img_path)
        batch["cond"][i].save(cond_img_path)
        # Replace 'synthetic'/'real' with artifacted since it was generated
        prompt = " ".join(["a", "artifacted"] + batch["text"][i].split(" ")[2:])
        # Adding the new lines to json file
        out_data.append(
            {
                "image": f"images/image{starting_idx+i}.png",
                "conditioning_image": f"conditioning_images/cond_image{starting_idx+i}.png",
                "text": prompt,
            }
        )

    return out_data, starting_idx + len(batch["text"])


def process_dataset(args):
    """Process new synthetic datasets by constructing conditioning images, generating new images and concatenating to existing train dataset

    Args:
        args (Parser.args): Processing configuration parameters
    """

    starting_idx = 0
    # Verifying if the output dataset dir exists and finding the number of images inside
    if path.isdir(args.out_dataset_folder):
        if "images" in os.listdir(args.out_dataset_folder):
            pth = path.join(args.out_dataset_folder, "images")
            starting_idx = len(os.listdir(pth))
        else:
            # Creating new folders in case they don't exist
            os.mkdir(path.join(args.out_dataset_folder, "images"))
            os.mkdir(path.join(args.out_dataset_folder, "conditioning_image"))
    else:
        # Creating new folders in case they don't exist
        os.mkdir(args.out_dataset_folder)
        os.mkdir(path.join(args.out_dataset_folder, "images"))
        os.mkdir(path.join(args.out_dataset_folder, "conditioning_image"))

    # Reading json data of the new dataset to transform
    in_data = read_json_data(args.in_json_file)
    # Reading json data from the existing train dataset (that will be extended)
    out_data = read_json_data(args.out_json_file)

    # Constructing SDXL+ControlNet pipeline
    pipe = load_pipeline(args)

    # Init batch
    batch = {"im": [], "cond": [], "text": []}
    for i in range(len(in_data)):
        # Constructing conditioning image (segmenatation + Canny)
        conditioning_image, trans_conditioning_image = construct_conditioning_image(
            in_data[i]["image"], in_data[i]["conditioning_image"]
        )
        batch["im"].append(trans_conditioning_image)
        # Using 'real' for inference
        prompt = " ".join(["a", "real"] + in_data[i]["text"].split(" ")[2:])
        batch["text"].append(prompt)
        batch["cond"].append(conditioning_image)
        if not (i % 3) and i:
            # Generate augmented images
            aug_images = transform_img(pipe, batch)
            batch["im"] = aug_images
            # Save generated images and adding lines to the train dataset json file
            out_data, starting_idx = save_batch(args, out_data, batch, starting_idx)
            batch = {"im": [], "cond": [], "text": []}

    # Save the concatenated version of the train dataset json file
    write_json_data(args.out_json_file, out_data)


if __name__ == "__main__":

    torch.cuda.empty_cache()
    args = parse_args()
    # 3, 2, 1 go ! ^^
    process_dataset(args)

    # Example Command :
    # python prepare_new_dataset.py --in_json_file="/data/cityscape/syn_city/images03/data_captionned.json" --out_json_file="/data/cityscapes/train.jsonl" --out_dataset_folder="/data/cityscapes" --unet_path="/out/controlnet_sdxl_unlocked_attention/checkpoint-5000/" --controlnet_path="/out/controlnet_sdxl_active_round1/"
