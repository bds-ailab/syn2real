import sys
import os
from controlnet_finetuning.dataset import MyDataset, CityDataset
from controlnet_finetuning.train import RESUME_PATH, MODEL_PATH, DATA_PATH, VAL_DATA_PATH

# Adding the src directory to the sys.path to ensure imports work correctly
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../ControlNet/ControlNet")
    ),
)

from share import *
import config

import matplotlib.pyplot as plt
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def eval(
    image,
    control,
    prompt,
    model,
    ddim_sampler,
    num_samples=3,
    strength=0.7,
    guess_mode=False,
    ddim_steps=50,
    seed=-1,
):
    """Evaluates the model on synthetic images by transforming them to realistic images. To do so we change the text caption
    form 'a synthetic picture ...' to 'a real picture ...'

    Args:
        image (ndarray): input synthetic image
        control (ndarray): control image (segmentation maps & canny edges)
        prompt (str): text prompt
        model (nn.Module): trained model
        ddim_sampler (nn.Module): Denoising Diffusion Implicit Model
        num_samples (int, optional): number of samples to generate. Defaults to 3.
        strength (float, optional): hyperparameter of generation diversity. Defaults to 0.7.
        guess_mode (bool, optional): generate output only from control image without text. Defaults to False.
        ddim_steps (int, optional): sampling steps for ddim. Defaults to 50.
        seed (int, optional): random seed. Defaults to -1.

    """

    with torch.no_grad():

        input_image = np.copy(image)
        detected_map = np.copy(control)
        # Reading input image resolution
        H, W, _ = input_image.shape

        # attaching data to cuda
        control = (torch.from_numpy(detected_map.copy()).float()).cuda()
        # Formatting control images to the number of desired samples
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        # Fixing seed for reproducibility
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # model.low_vram_shift(is_diffusing=False)

        # Concatenating conditioning embeddings and control images
        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        # model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # "Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01" - original implementation

        # Sampling !
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=0,
            unconditional_guidance_scale=9,
        )

        # model.low_vram_shift(is_diffusing=False)

        # Decoding output
        x_samples = model.decode_first_stage(samples)
        # Formatting to readable format
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )
        results = [x_samples[i] for i in range(num_samples)]

    return [detected_map] + results


if __name__ == "__main__":

    # Creating model from yaml file
    model = create_model("/models/cldm_v15.yaml").cpu()
    # Loading trained weights
    model.load_state_dict(
        load_state_dict(
            "/models/controlnet_canny_unlocked_ep30.ckpt",
            location="cuda",
        )
    )
    # Attaching model to cuda
    model = model.cuda()
    # Initializing DDIM with constucted model
    ddim_sampler = DDIMSampler(model)

    # Loading validation dataset
    val_dataset = CityDataset(
        data_path=VAL_DATA_PATH, shape=(512, 256), canny=True, noise=False, val=True
    )

    # Processing each image in dataset
    for idx in range(40, 100):
        print("==============")
        print(f"Evaluating example {idx}")
        print("==============")

        example = val_dataset[idx]

        results = eval(
            example["jpg"],
            example["hint"],
            example["txt"],
            model,
            ddim_sampler,
            num_samples=3,
            strength=1,
            guess_mode=False,
            ddim_steps=50,
        )

        # Saving results in out folder

        im = cv2.cvtColor((example["jpg"] + 1) / 2, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"/out/controlnet_val_out/target_{idx}.png", (im * 255).astype(int))
        im = cv2.cvtColor(example["hint"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"/out/controlnet_val_out/source_{idx}.png", (im * 255).astype(int))

        for i in range(1, len(results)):
            im = cv2.cvtColor(results[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"/out/controlnet_val_out/out{i}_{idx}.png", im)
