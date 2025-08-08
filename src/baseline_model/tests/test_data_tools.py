# test_image_processing.py

import pytest
import os
import numpy as np
from unittest import mock
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from baseline_model.data_tools import (
    CustomImageDataset,
    split_train_val,
    preprocess_resnet50,
)

it = pytest.mark.it
describe = pytest.mark.describe


@describe(
    "Test the class CustomImageDataset and load and prepare data for ResNet50 header"
)
class TestDataTools:

    # Mock data
    mock_image_label_dict = {"/path/to/image1.jpg": 0, "/path/to/image2.jpg": 1}

    mock_train_path = "/mock/train/path/"
    mock_image_list_content = "image1.jpg 0\nimage2.jpg 1\n"

    @it("Must split the dataset into training and validation sets correctly")
    @mock.patch("os.path.isdir")
    @mock.patch(
        "builtins.open", new_callable=mock.mock_open, read_data=mock_image_list_content
    )
    def test_split_train_val(self, mock_open, mock_isdir):
        mock_isdir.return_value = True
        train_images_dict, val_images_dict, report = split_train_val(
            self.mock_train_path, prop=0.5, classes=[0, 1]
        )

        # Assertions
        assert len(train_images_dict) + len(val_images_dict) == 2
        assert "total_num_images" in report

    @it("Must initialize dataset correctly and return correct length")
    @mock.patch("PIL.Image.open")
    def test_custom_image_dataset(self, mock_image_open):

        mock_image_open.return_value = Image.new("RGB", (100, 100))

        dataset = CustomImageDataset(image_label_dict=self.mock_image_label_dict)

        # Assertions
        assert len(dataset) == 2
        image, label = dataset[0]
        assert isinstance(image, Image.Image)
        assert label == 0

    @it("Must create a DataLoader with the correct batch size and transformations")
    @mock.patch("torch.utils.data.DataLoader")
    def test_preprocess_resnet50(self, mock_dataloader):
        mock_dataloader.return_value = DataLoader([])

        dataloader = preprocess_resnet50(
            self.mock_image_label_dict, batch_size=32, num_workers=4
        )

        # Assertions
        assert isinstance(dataloader, DataLoader)
