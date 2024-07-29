import os
import random


import numpy as np
import torch
import torch.utils.checkpoint
from transformers import PretrainedConfig

from diffusers.utils import make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    """import the right model class of the passed model name/path

    Args:
        pretrained_model_name_or_path (str): model name (hub) or path (local)
        revision (str): model revision
        subfolder (str, optional): model subfolder. Defaults to "text_encoder".

    Raises:
        ValueError: model class is not supported if different from (CLIPTextModel, CLIPTextModelWithProjection)

    Returns:
        type: the text embedding model class
    """
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(
                os.path.join(repo_folder, f"images_{i}.png")
            )
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
    # controlnet-{repo_id}

    These are controlnet weights trained on {base_model} with new type of conditioning.
    {img_str}
    """

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(
    prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True
):
    """encode the text prompts

    Args:
        prompt_batch (list): list/batch of prompts
        text_encoders (Module): encoding models
        tokenizers (Module): tocknizing models
        proportion_empty_prompts (float): proportion of empty prompts
        is_train (bool, optional): is the batch for training or validation. Defaults to True.

    Returns:
        torch.Tensor: prompts embeddings
        torch.Tensor: pooled prompts embeddings
        syn_or_real: batch bool values for each caption if synthetic or real
    """
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        # Remove some captions et leave only the syn/real tocken
        if random.random() < proportion_empty_prompts:
            token = ""
            if "synthetic" in caption:
                token = "a synthetic picture"
            else:
                token = "a real picture"
            captions.append(token)
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    # Embedding texts
    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    syn_or_real = ["synthetic" in caption for caption in prompt_batch]
    return prompt_embeds, pooled_prompt_embeds, syn_or_real
