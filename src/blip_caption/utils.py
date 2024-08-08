from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import torch
from config import DEVICE, DATA_PATH, FILE_NAME, PROMPT
import json


def load_model(device=DEVICE):
    """loads BLIP model and input/output processor

    Args:
        device (str, optional): computing device (cuda for acceleration). Defaults to DEVICE.

    Returns:
        (transformers.models, transformers.models): I/O processor and BLIP model
    """
    # Loading BLIP processor from pretrained model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Loading BLIP model from pretrained model
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    return processor, model


def load_data(data_path=DATA_PATH):
    """loads data from given dataset path

    Args:
        data_path (str, optional): path to dataset with images to be captioned. Defaults to DATA_PATH.

    Returns:
        paths (list): list of images absolute paths
    """
    with open(data_path, "r") as openfile:
        # Reading from json file
        paths = json.load(openfile)

    return paths


def process(processor, model, paths, prompt=PROMPT):
    """Process images and generate captions

    Args:
        processor (transformers.models): BLIP processor
        model (transformers.models): BLIP model
        paths (paths): list of abs paths to images

    Returns:
        captions (list): list of generated captions
    """
    i = 0
    captions = []

    # Iterating over dataset images
    while i < len(paths):

        # Loading images batch by batch for memory efficiency
        images = [
            Image.open(path["image"]).convert("RGB") for path in paths[i : i + 20]
        ]

        # Starting the text prompt with 'a picture of '
        text = [prompt] * len(images)

        # Preprocessing the images and texts for BLIP
        # Note: batch is attached to GPU and using float16 for acceleration
        inputs = processor(images, text, return_tensors="pt").to(DEVICE, torch.float16)

        # Generate captions using BLIP
        out = model.generate(**inputs)

        # Decoding the generated output using the Processor
        captions += [
            processor.decode(out[i], skip_special_tokens=True)
            for i in range(len(images))
        ]

        # Printing progress
        if i + 20 < len(paths):
            print(f"Processed : {i+20}/{len(paths)}")
        else:
            print(f"Processed : {len(paths)}/{len(paths)}")
        i += 20

    return captions


def save_captions(paths, captions, filename=FILE_NAME, prompt=PROMPT):
    """save captions with images paths in a txt file

    Args:
        paths (list): list of images abs paths
        captions (list): list of generated captions
        filename (str, optional): txt file where to save the data. Defaults to FILE_NAME.
        special_token (str, optional): a special token to add at the start of the caption for all images. Defaults to TOKEN.
    """
    assert len(paths) == len(captions)
    for i in range(len(paths)):
        # The paths[i]["dataset"] can be 'real', 'synthetic' or 'artifacted
        paths[i]["text"] = (
            f"a {paths[i]['dataset']} picture of " + captions[i][len(prompt) :]
        )

    with open(filename, "w") as f:
        json.dump(paths, f)
