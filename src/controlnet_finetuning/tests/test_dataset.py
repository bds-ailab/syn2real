import cv2
import numpy as np
import pytest
from unittest import mock
from torch.utils.data import Dataset
from controlnet_finetuning.dataset import (
    MyDataset,
    CityDataset,
)  # Adjust the import according to your actual module name

it = pytest.mark.it
describe = pytest.mark.describe


@describe("Test the functions of the class MyDataset")
class TestMyDataset:
    @it("must correctly initialize and load data")
    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data='{"source": "source_image.jpg", "target": "target_image.jpg", "prompt": "A prompt"}\n',
    )
    def test_init(self, mock_open):
        dataset = MyDataset(data_path="path/to/data/")
        assert len(dataset) == 1

    @it("must correctly fetch an item by index")
    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data='{"source": "source_image.jpg", "target": "target_image.jpg", "prompt": "A prompt"}\n',
    )
    @mock.patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8))
    def test_getitem(self, mock_open, mock_imread):
        dataset = MyDataset(data_path="path/to/data/")
        item = dataset[0]
        assert item["txt"] == "A prompt"
        assert item["hint"].shape == (256, 256, 3)
        assert item["jpg"].shape == (256, 256, 3)


@describe("Test the functions of the class CityDataset")
class TestCityDataset:
    @it("must correctly initialize and load data")
    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data='[{"source": "source_image.jpg", "target": "target_image.jpg", "caption": "A caption"}]',
    )
    def test_init(self, mock_open):
        dataset = CityDataset(data_path="path/to/data/")
        assert len(dataset) == 1

    @it("must correctly fetch an item by index and apply data augmentation")
    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data='[{"source": "source_image.jpg", "target": "target_image.jpg", "caption": "A caption"}]',
    )
    @mock.patch("cv2.imread", return_value=np.zeros((1024, 512, 3), dtype=np.uint8))
    def test_getitem(self, mock_open, mock_imread):
        dataset = CityDataset(
            data_path="path/to/data/", shape=(1024, 512), canny=True, noise=True
        )
        item = dataset[0]
        assert item["txt"] == "A caption"
        assert item["hint"].shape == (512, 1024, 3)
        assert item["jpg"].shape == (512, 1024, 3)
