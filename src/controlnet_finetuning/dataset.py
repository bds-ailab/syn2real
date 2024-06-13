import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from config import DATA_PATH


class MyDataset(Dataset):
    def __init__(self, data_path=DATA_PATH):
        """function: Dataset fetching class

        Args:
            data_path (str, optional): path to training dataset. Defaults to DATA_PATH.
        """
        self.data = []
        self.data_path = data_path

        # Reading text prompts for the images
        with open(data_path + "prompt.json", "rt") as f:
            for line in f:
                # Each line containes the text prompt, the path to target image and the path to control (source) image
                self.data.append(json.loads(line))

    def __len__(self):
        """function: returns the number of instances in Dataset

        Returns:
            (int): Dataset size
        """
        return len(self.data)

    def __getitem__(self, idx):
        """function: Gets one image from the dataset by index

        Args:
            idx (int): the image index in the dataset
        Returns:
            (dict): dictionnary of target image, text prompt and source image
        """
        # Selecting the idx-item of the dataset
        item = self.data[idx]

        # Extracting source image (control image)
        source_filename = item["source"]
        # Extracting target image (output image)
        target_filename = item["target"]
        # Extracting text prompt
        prompt = item["prompt"]

        # Resizing images to fit more in GPUs memory for each batch
        source = cv2.resize((cv2.imread(self.data_path + source_filename)), (256, 256))
        target = cv2.resize(cv2.imread(self.data_path + target_filename), (256, 256))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
