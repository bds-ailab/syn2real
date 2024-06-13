import sys
import os
from dataset import MyDataset
from train_config import RESUME_PATH, MODEL_PATH

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


def train():
    """function: runs the finetuning pipeline by loading the data, loading the model and lunching the trainer"""

    # Configs
    resume_path = RESUME_PATH
    # Training batch size
    batch_size = 10
    # Logger frequency to save images
    logger_freq = 300
    # Training learning rate
    learning_rate = 4e-5
    # Freezing Stable diffusion weights
    sd_locked = True
    # Injecting control in every block of the decoder and not just mid block
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(MODEL_PATH).cpu()
    model.load_state_dict(load_state_dict(resume_path, location="cpu"))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Loading dataset and applying necessary transforms
    dataset = MyDataset()

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
    trainer = pl.Trainer(
        devices=2,
        accelerator="gpu",
        precision=16,
        callbacks=[logger],
        strategy="ddp",
        accumulate_grad_batches=2,
    )

    # Train!
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    train()
