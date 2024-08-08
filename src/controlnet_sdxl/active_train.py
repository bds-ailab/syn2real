from accelerate.utils import write_basic_config
import subprocess
import yaml
from os import path
import os

# Write accelerate configuration for ressources usage
write_basic_config()


def read_train_config():
    """Read configuration parameters from config.yml file in current directory

    Returns:
        dict: Dictionnary of configuration parameters
    """
    config_path = "config.yml"
    # Open yml file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def controlnet_train(
    base_model_dir, unet_dir, controlnet_dir, output_dir, lr, bs, max_steps, acc_steps
):
    """Launch controlnet training

    Args:
        base_model_dir (str): SDXL base model path
        unet_dir (str): unet weights path
        controlnet_dir (str): controlnet weights path
        output_dir (str): path to save checkpoints
        lr (float): learning rate
        bs (int): batch size
        max_steps (int): maximum number of training steps
        acc_steps (int): number of batch gradient accumulation

    Returns:
        int: returned code after executing the command
    """
    command = f"accelerate launch /sdxl/train_controlnet_sdxl.py --controlnet_model_name_or_path={controlnet_dir} --unet_model_name_or_path={unet_dir} --pretrained_model_name_or_path={base_model_dir}  --output_dir={output_dir}  --train_data_dir=/data/cityscapes --mixed_precision='fp16'  --resolution=1024 --report_to='wandb'  --learning_rate={lr}  --max_train_steps={max_steps}  --train_batch_size={bs}  --gradient_accumulation_steps={acc_steps}  --seed=42 --use_8bit_adam --set_grads_to_none --dataloader_num_workers=25 --proportion_empty_prompts=0.2 --canny_edges"
    p = subprocess.Popen(command, shell=True)
    return p.wait()


def unet_train(
    base_model_dir, unet_dir, controlnet_dir, output_dir, lr, bs, max_steps, acc_steps
):
    """Launch unet attention layers training

    Args:
        base_model_dir (str): SDXL base model path
        unet_dir (str): unet weights path
        controlnet_dir (str): controlnet weights path
        output_dir (str): path to save checkpoints
        lr (float): learning rate
        bs (int): batch size
        max_steps (int): maximum number of training steps
        acc_steps (int): number of batch gradient accumulation

    Returns:
        int: returned code after executing the command
    """
    command = f"accelerate launch /sdxl/train_sdxl_unlocked.py --controlnet_model_name_or_path={controlnet_dir} --unet_model_name_or_path={unet_dir} --pretrained_model_name_or_path={base_model_dir}  --output_dir={output_dir}  --train_data_dir=/data/cityscapes --mixed_precision='fp16'  --resolution=1024 --report_to='wandb'  --learning_rate={lr}  --max_train_steps={max_steps}  --train_batch_size={bs}  --gradient_accumulation_steps={acc_steps}  --seed=42 --use_8bit_adam --set_grads_to_none --dataloader_num_workers=25 --proportion_empty_prompts=0.2 --canny_edges"
    p = subprocess.Popen(command, shell=True)
    return p.wait()


def prepare_data(in_json, out_json, unet_path, controlnet_path):
    """Prepare new dataset for next training round

    Args:
        in_json (str): path to new synthetic dataset json file
        out_json (str): path to existing train dataset json file
        unet_path (str): path to unet pretrained weigths
        controlnet_path (str): path to controlnet pretrained weights

    Returns:
        int: returned code after executing the command
    """
    command = f"python prepare_new_dataset.py --in_json_file={in_json} --out_json_file={out_json} --out_dataset_folder='/data/cityscapes' --unet_path={unet_path} --controlnet_path={controlnet_path}"
    p = subprocess.Popen(command, shell=True)
    return p.wait()


def active_train_loop(config):
    """Launch active learning train loop

    Args:
        config (dict): Dictionnary of configuration parameters
    """

    for r in range(config["nb_rounds"]):
        print(
            f"################ Round {r} of active learning started ##################"
        )
        if r > 0:
            # Transform new datasets and add them to training dataset
            prepare_data(
                config[f"round{r}"]["in_json"],
                config[f"round{r}"]["out_json"],
                config[f"round{r}"]["unet_path"],
                config[f"round{r}"]["controlnet_path"],
            )

        if not path.isdir(config[f"round{r}"]["output_dir"]):
            os.mkdir(config[f"round{r}"]["output_dir"])

        # Train ControlNet Weights First
        controlnet_train(
            config[f"round{r}"]["base_model_path"],
            config[f"round{r}"]["unet_path"],
            config[f"round{r}"]["controlnet_path"],
            config[f"round{r}"]["output_dir"],
            config[f"round{r}"]["lr"],
            config[f"round{r}"]["bs"],
            config[f"round{r}"]["max_steps"],
            config[f"round{r}"]["acc_steps"],
        )

        if not path.isdir(config[f"round{r}"]["output_dir2"]):
            os.mkdir(config[f"round{r}"]["output_dir2"])

        # Now Freeze ControlNet and Train unlocked attention layers of Unet
        unet_train(
            config[f"round{r}"]["base_model_dir2"],
            config[f"round{r}"]["unet_dir2"],
            config[f"round{r}"]["controlnet_dir2"],
            config[f"round{r}"]["output_dir2"],
            config[f"round{r}"]["lr2"],
            config[f"round{r}"]["bs2"],
            config[f"round{r}"]["max_steps2"],
            config[f"round{r}"]["acc_steps2"],
        )


if __name__ == "__main__":

    # Read train configuration
    config = read_train_config()
    # Launch training rounds
    active_train_loop(config)
