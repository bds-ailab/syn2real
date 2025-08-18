from PIL import Image
import numpy as np
from IPython.display import clear_output
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from baseline_model.config import SR, BATCH_SIZE, NUM_WORKERS


class CustomImageDataset(Dataset):
    def __init__(self, image_label_dict, transform=None):
        """function: Custom torch dataset object to feed the dataloader

        Args:
            image_label_dict (dict): images dictionnary with the format {img_path:label}
            transform (torchvision.transforms.transforms.Compose, optional): image preprocessing transformations. Defaults to None.
        """
        # Iniating class attributes
        self.image_label_dict = image_label_dict
        # Extracting the images paths
        self.image_paths = list(image_label_dict.keys())
        # Possible transformations
        self.transform = transform

    def __len__(self):
        """function: returns the number of instances in Dataset

        Returns:
            (int): Dataset size
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """function: Gets one image from the dataset by index

        Args:
            idx (int): the image index in image_paths

        Returns:
            image (PIL image): RGB image in the image_paths[idx]
            label (int): The class label of that image
        """
        # Extracting image path
        img_path = self.image_paths[idx]

        # Loading the image in RGB format
        image = Image.open(img_path).convert("RGB")

        # Extracting the image label
        label = self.image_label_dict[img_path]

        # Applying necessary transforms (in case they exist)
        if self.transform:
            image = self.transform(image)

        return image, label


def split_train_val(train_path, prop=SR, classes=list(range(12))):
    """function: splits training dataset into training and validation

    Args:
        train_path (str): path to train images list (with labels) file (.txt)
        prop (float, optional): proportion of validation set 0-1. Defaults to 0.15.
        classes (list, optional): list classes to take into consideration. Defaults to list(range(12)).

    Returns:
        train_images_dict (dict): dictionnary of training images (format {img_path:label})
        val_images_dict (dict): dictionnary of training images (format {img_path:label})
        report (dict): dictionnay of metadata about the train/val datasets
    """

    if prop > 1 or prop < 0:
        raise ValueError("split proportion must be 0<=prop<=1")
    if not os.path.isdir(train_path):
        raise ValueError("train dataset path not found")

    # Initiating train/val dictionnaries
    train_images_dict = {}
    val_images_dict = {}

    # Init report to save metadata of each dataset
    report = {}

    # Total number of images
    num_images = 0

    # Opening the train image_list file that contains the label of each synthetic image
    with open(train_path + "image_list.txt", "r") as f:

        # Reading the files lines
        lines = f.readlines()
        num_images = len(lines)
        num_classes = 0
        class_names = []

        for i in range(len(lines)):
            # Reading the file name and label (int) of each image
            name, id = lines[i].strip().split(" ")

            # The class name is the first part of its relative path exp : aeroplane/imgxxx.jpg
            class_name = name.split("/")[0]

            # Checking if the class is wanted by the user
            if int(id) in classes:

                # Sampling from uniform distribution U(0, 1) to decide train or val
                # Note : the obtained number of validation instances is ~= 0.15*num_images but not exact
                n = np.random.rand()
                if n > prop:
                    # Adding the image to train set (with absolute path)
                    train_images_dict[train_path + name] = int(id)

                    # Counting the number of instances per class for train set
                    if f"num_{class_name}_train" in report:
                        report[f"num_{class_name}_train"] += 1
                    else:
                        report[f"num_{class_name}_train"] = 0
                        num_classes += 1
                        class_names.append(class_name)
                else:
                    # Adding the image to train set (with absolute path)
                    val_images_dict[train_path + name] = int(id)

                    # Counting the number of instances per class for val set
                    if f"num_{class_name}_val" in report:
                        report[f"num_{class_name}_val"] += 1
                    else:
                        report[f"num_{class_name}_val"] = 0
            else:
                # Not counting the images of unwanted classes
                num_images -= 1

        # Adding metadata infos to the report
        report["total_num_images"] = num_images
        report["train_num_images"] = len(train_images_dict)
        report["val_num_images"] = len(val_images_dict)
        report["num_classes"] = num_classes
        report["class_names"] = class_names

    return train_images_dict, val_images_dict, report


def preprocess_resnet50(images_dict, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """function: Load and prepare synthetic data for ResNet50 header format
    Args:
        images_dict (dict): images paths and labels
        batch_size (int, optional): training batch size. Defaults to BATCH_SIZE.
        num_workers (int, optional): number of workers to load batches data. Defaults to NUM_WORKERS.

    Returns:
        train_dataloader (torch.Dataloader): training dataloader
        val_dataloader (torch.Dataloader): val dataloader
        report (dict): report of metadata
    """

    # Defining Necessary transformations for ResNet50
    # Images were resized using aspect ratio to maintain aspect ratio
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomGrayscale(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ]
    )

    # Create the train dataset
    dataset = CustomImageDataset(image_label_dict=images_dict, transform=transform)

    # Create the train DataLoader
    # Note: num_workers will accelerate loading data from disk to GPU
    # Note: pin_memory is enabled to allow non_blocking transfer (acceleration reasons)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
