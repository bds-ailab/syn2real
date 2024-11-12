from segment_anything import build_sam_vit_h, SamPredictor, SamAutomaticMaskGenerator
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
from time import time
import argparse
import os
import json


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


def parse_args():

    parser = argparse.ArgumentParser(
        description="Construct semantic segmentation using SAM"
    )

    parser.add_argument(
        "--in_dataset",
        type=str,
        required=True,
        help="path to input dataset",
    )

    parser.add_argument(
        "--out_dataset",
        type=str,
        required=True,
        help="path to output dataset",
    )

    parser.add_argument(
        "--start_idx",
        type=int,
        required=False,
        default=0,
        help="starting index for the dataset fragment",
    )

    parser.add_argument(
        "--end_idx",
        type=int,
        required=False,
        default=0,
        help="end index for the dataset fragment",
    )

    args = parser.parse_args()
    return args


def construct_seg_model():
    """creates SAM segmentation model from hugginface hub checkpoints

    Returns:
        SamAutomaticMaskGenerator: SamAutomaticMaskGenerator
    """
    chkpt_path = hf_hub_download(
        "ybelkada/segment-anything", "checkpoints/sam_vit_h_4b8939.pth"
    )
    sam = build_sam_vit_h(checkpoint=chkpt_path)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def generate_mask(image_pth, mask_generator):
    """Generates semantic segmentation mask for a given image using SAM

    Args:
        image_pth (str): path to the input image
        mask_generator (SamAutomaticMaskGenerator): semantic segmenter

    Returns:
        list: list of masks and annotations
    """
    image = cv2.imread(image_pth)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)
    return masks


def merge_masks(masks):
    """merge the predicted masks by SAM

    Args:
        masks (list): list of predicted masks

    Returns:
        ndarray: merged image of all the masks
    """
    size = masks[0]["segmentation"].shape
    global_mask = np.zeros((size[0], size[1], 3))
    for m in masks:
        color = np.random.randint(0, 255, 3)
        global_mask[m["segmentation"]] = color
    return global_mask


def main(args):
    """predicts semantic segmentation from a given dataset and save it in an output folder

    Args:
        args : Parsed arguments
    """
    # Create output folder if do
    if not os.path.isdir(args.out_dataset):
        os.mkdir(args.out_dataset)
    # Construct the segmentation model from SAM checkpoints
    mask_generator = construct_seg_model()
    # Read the input dataset images paths
    data = read_json_data(args.in_dataset)
    # Restrict the transformed portion of the dataset
    if not args.end_idx:
        end_idx = len(data)
    else:
        end_idx = args.end_idx
    data = data[args.start_idx : end_idx]
    for line in data:
        # Read
        img_pth = line["conditioning_image"]
        # Predict
        masks = generate_mask(img_pth, mask_generator)
        # Merge
        global_mask = merge_masks(masks)
        # Save
        filename = img_pth.split("/")[-1]
        cv2.imwrite(os.path.join(args.out_dataset, filename), global_mask)


if __name__ == "__main__":
    args = parse_args()
    main(args)
