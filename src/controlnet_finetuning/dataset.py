import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_path):
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


class CityDataset(Dataset):
    def __init__(
        self, data_path, shape=(1024, 512), canny=False, noise=True, val=False
    ):
        """function: Dataset fetching class

        Args:
            data_path (str, optional): path to training dataset. Defaults to DATA_PATH.
        """
        self.data_path = data_path
        self.shape = shape
        self.canny = canny
        self.noise = noise
        self.val = val

        # Reading text prompts for the images
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        """function: returns the number of instances in Dataset

        Returns:
            (int): Dataset size
        """
        return len(self.data)

    def __getitem__(self, idx):
        """function: Gets one image from the dataset by index and apply data augmentation techniques

        Args:
            idx (int): the image index in the dataset
        Returns:
            (dict): dictionnary of target image, text prompt and source image
        """

        # IMPORTANT : The data augmentation implemented in this loading function aim to help
        # The model to generalize better and learn the meaning of each part of the control image
        # without overfitting on a certain unintentially repeated pattern in the data

        # Selecting the idx-item of the dataset
        item = self.data[idx]

        # Extracting source image (control image)
        source_filename = item["source"]
        # Extracting target image (output image)
        target_filename = item["target"]

        # Extracting text prompt
        prompt = item["caption"]

        # Loading images from paths
        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Removing the segmentation maps for some images and leaving only canny edges for control
        # Will help the model understand the information from canny edges alone
        seg_removed = False
        # self.val variable will prevent applying augmentation ops during inference or validation
        if ("syn_city" in item["source"]) and not (self.val):
            n = np.random.rand()
            # Applying this op on half of synthetic dataset since half of real dataset already
            # does not have segmentation maps (only canny)
            if n < 0.5:
                source[:, :, :] = 0
                seg_removed = True

        # Add canny contours to the control image
        if self.canny:
            n = np.random.rand()
            if (
                n < 0.2
                and not ("test" in item["source"])
                and not (seg_removed)
                and not (self.val)
            ):
                # Not applying canny transformations for 20% of the dataset to leave just seg maps
                pass
            else:
                # Different thresholds for real in syn images because of the noise present in syn images
                if "real_city" in item["source"]:
                    low_threshold = 50
                    high_threshold = 120

                # Thresholds were selected experimentaly by looking for values that generates the same output
                # for both real and synthetic images
                else:
                    low_threshold = 150
                    high_threshold = 200

                # Transforming the target image to canny edges
                canny_image = cv2.Canny(target, low_threshold, high_threshold)

                # Drawing thicker and more visible line with dilate
                kernel = np.ones((2, 2), np.uint8)
                canny_image = cv2.dilate(canny_image, kernel, iterations=1)
                canny_image = canny_image[:, :, None]

                # Formatting and normalizing the canny edges image to the same shape of source images
                canny_image = (
                    np.concatenate([canny_image, canny_image, canny_image], axis=2)
                    / 255
                )

                # Combining segmentation map with canny edges
                source[canny_image == 1] = 255

        # Resizing images
        source = cv2.resize(source, self.shape)
        target = cv2.resize(target, self.shape)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Adding noise to prevent overfitting (not used in last experimentations)
        if self.noise:
            black_noise = (np.random.normal(0, 1, source.shape) < 0.6).astype(int)
            source *= np.uint8(black_noise)

        # Replacing some segments with black mask for the model to guess from shape
        # Will help the model generate plausible outputs from imperfect segmentations
        if "real_city" in item["source"]:
            pth = item["source"][:-9] + "labelIds.png"
            im_ids = cv2.imread(pth)
            im_ids = cv2.resize(im_ids, self.shape)
            class_id = np.random.randint(1, 33, 4)
            for id in class_id:
                source[im_ids == id * np.ones(3)] = 0

        # Eliminating the BLIP description for 20% of the images
        n = np.random.rand()
        if not (self.val):
            if n < 0.2:
                # Leaving the information about real/synthetic
                if "real" in prompt:
                    prompt = "a real picture"
                else:
                    prompt = "a synthetic picture"

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
