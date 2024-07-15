import sys
import os
from dataset import MyDataset, CityDataset
import torch
import argparse
import yaml
import subprocess

# Adding the src directory to the sys.path to ensure imports work correctly
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../ControlNet/ControlNet")
    ),
)


from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


def train(
    resume_path,
    model_path,
    data_path,
    batch_size=1,
    logger_freq=100,
    learning_rate=1e-5,
    sd_locked=True,
    only_mid_control=False,
    shape=(512, 256),
    canny=True,
    noise=False,
    accumulate_grad_batches=10,
    num_devices=2,
    max_epochs=50,
    precision=32,
):
    """runs the finetuning pipeline by loading the data, loading the model and lunching the trainer

    Args:
        resume_path (str): weights file path of the initial checkpoint or other resume checkpoint (ckpt file)
        model_path (str): model architecture file path (yml file)
        data_path (str): dataset description file path (json file)
        batch_size (int, optional): training batch size. Defaults to 1.
        logger_freq (int, optional): image logging frequency. Defaults to 100.
        learning_rate (float, optional): learning rate. Defaults to 1e-5.
        sd_locked (bool, optional): lock the decoder layers of SD base. Defaults to True.
        only_mid_control (bool, optional): add control only in the middle SD block. Defaults to False.
        shape (tuple, optional): training images shape. Defaults to (512, 256).
        canny (bool, optional): add canny edges to control images. Defaults to True.
        noise (bool, optional): add noise to control images. Defaults to False.
        accumulate_grad_batches (int, optional): number of batches gradient to accumulate. Defaults to 10.
        num_devices (int, optional): number of computing devices. Defaults to 2.
        max_epochs (int, optional): max number of epochs. Defaults to 50.
    """

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_path).cpu()
    model.load_state_dict(load_state_dict(resume_path, location="cpu"))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Loading dataset and applying necessary transforms
    # The images are resized to low resolution for faster training
    dataset = CityDataset(data_path=data_path, shape=shape, canny=canny, noise=noise)

    # Defining the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=50,
        persistent_workers=True,
    )

    # Defining the image logger to supervise the training
    logger = ImageLogger(batch_frequency=logger_freq)

    # Defining pytorch trainer with multiple gpus strategy
    # Note: accumulate grad will accumulate the gradient of multiple batches before updating
    # This parameter simulates large batch sizes which leads to more stable learning
    trainer = pl.Trainer(
        devices=num_devices,
        accelerator="gpu",
        precision=precision,
        callbacks=[logger],
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=max_epochs,
    )

    # Train!
    trainer.fit(model, dataloader)


def main():
    """Parse training config file and save experiment with its parameters and weights files to be logged in mlflow"""

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

    # Specify if the experiment should be logged in mlflow
    parser.add_argument(
        "--to_log",
        type=bool,
        default=True,
        help="Bool variable to log the experiment or not",
    )

    args = parser.parse_args()

    # Read yaml configuration file
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    # assemble the experiment resume in a yaml file to be used by mlflow module later
    experiment_yaml = {
        "exp_name": config["exp_config"]["exp_name"],
        "run_name": config["exp_config"]["run_name"],
        "weights_file": f'/models/{config["exp_config"]["exp_name"]}_{config["exp_config"]["run_name"]}/trained_model_weights.ckpt',
        "config_file": f'/out/{config["exp_config"]["exp_name"]}_{config["exp_config"]["run_name"]}/config.yml',
        "to_log": args.to_log,
        "logged": False,
    }

    # Empty cuda cache to prevent 'some' CUDA out of memory errors
    torch.cuda.empty_cache()
    # Lunch training with specified parameters
    train(
        resume_path=config["model_config"]["resume_path"],
        model_path=config["model_config"]["model_path"],
        data_path=config["data_config"]["data_path"],
        batch_size=config["train_config"]["batch_size"],
        logger_freq=config["train_config"]["logger_freq"],
        learning_rate=float(config["train_config"]["learning_rate"]),
        sd_locked=config["model_config"]["sd_locked"],
        only_mid_control=config["model_config"]["only_mid_control"],
        shape=(config["data_config"]["img_w"], config["data_config"]["img_h"]),
        canny=config["data_config"]["canny"],
        noise=config["data_config"]["noise"],
        accumulate_grad_batches=config["train_config"]["accumulate_grad_batches"],
        num_devices=config["train_config"]["num_devices"],
        max_epochs=config["train_config"]["max_epochs"],
        precision=config["train_config"]["precision"],
    )

    # Find last version (last epoch) checkpoint and extract weights path
    model_log_path = "./lightning_logs/"
    last_version = os.listdir(model_log_path)[-1]
    check_path = f"{model_log_path}/{last_version}/checkpoints/"
    trained_model_path = check_path + os.listdir(check_path)[0]

    # Create an experiment folder in models volume if it doesn't exist
    if not (
        f'{config["exp_config"]["exp_name"]}_{config["exp_config"]["run_name"]}'
        in os.listdir("/models/")
    ):

        mkdir_command = f'mkdir /models/{config["exp_config"]["exp_name"]}_{config["exp_config"]["run_name"]}/'
        p = subprocess.Popen(mkdir_command, stdout=subprocess.PIPE, shell=True)
        _ = p.wait()

    # Split and save the model on multiple smaller parts to be uploaded
    split_command = (
        f"split -b 500M {trained_model_path} {experiment_yaml['weights_file']}.part"
    )
    p = subprocess.Popen(split_command, stdout=subprocess.PIPE, shell=True)
    _ = p.wait()

    # Document if multiple parts were needed or only one
    num_parts = len(
        os.listdir(
            f'/models/{config["exp_config"]["exp_name"]}_{config["exp_config"]["run_name"]}/'
        )
    )
    if num_parts > 1:
        experiment_yaml["multiple_parts"] = True
    else:
        experiment_yaml["multiple_parts"] = False

    # Create an experiment folder in out volume if it doesn't exist
    if not (
        f'{config["exp_config"]["exp_name"]}_{config["exp_config"]["run_name"]}'
        in os.listdir("/out/")
    ):

        mkdir_command = f'mkdir /out/{config["exp_config"]["exp_name"]}_{config["exp_config"]["run_name"]}/'
        p = subprocess.Popen(mkdir_command, stdout=subprocess.PIPE, shell=True)
        _ = p.wait()

    # Save congiguration parameters
    save_config_command = f'cp {args.config_path} {experiment_yaml["config_file"]}'
    p = subprocess.Popen(save_config_command, stdout=subprocess.PIPE, shell=True)
    _ = p.wait()

    # Save experiment resume
    with open(
        f'/out/{config["exp_config"]["exp_name"]}_{config["exp_config"]["run_name"]}/exp.yml',
        "w",
    ) as yaml_file:
        yaml.dump(experiment_yaml, yaml_file, default_flow_style=False)


if __name__ == "__main__":
    main()
