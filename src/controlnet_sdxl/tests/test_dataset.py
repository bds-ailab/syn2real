import numpy as np
import pytest
from unittest import mock
from datasets import DatasetDict, Dataset
from controlnet_sdxl.dataset import get_train_dataset, prepare_train_dataset, collate_fn
from PIL import Image
import torch

it = pytest.mark.it
describe = pytest.mark.describe


@pytest.fixture
def mock_args():
    class MockArgs:
        dataset_name = "dataset_name"
        train_data_dir = None
        image_column = "image"
        caption_column = "caption"
        conditioning_image_column = "conditioning_image"
        cache_dir = "cache_dir"
        resolution = 512
        canny_edges = True
        seed = 42
        max_train_samples = None
        dataset_config_name = "config_name"

    return MockArgs()

@pytest.fixture
def mock_accelerator():
    class MockAccelerator:
        def main_process_first(self):
            class ContextManager:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    pass

            return ContextManager()

    return MockAccelerator()

@pytest.fixture
def mock_logger():
    class MockLogger:
        def info(self, message):
            pass

    return MockLogger()

@pytest.fixture
def mock_dataset():
    data = {
        "train": {
            "image": [Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))],
            "caption": ["A sample caption"],
            "conditioning_image": [Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))],
            "syn_or_real": [True],
        }
    }
    return DatasetDict({k: Dataset.from_dict(v) for k, v in data.items()})

@describe("Test the functions of the dataset handling")
class TestDatasetFunctions:
    
    @it("should correctly load the training dataset")
    @mock.patch("controlnet_sdxl.dataset.load_dataset")
    def test_get_train_dataset(self, mock_load_dataset, mock_args, mock_accelerator, mock_logger, mock_dataset):
        mock_load_dataset.return_value = mock_dataset
        
        dataset = get_train_dataset(mock_args, mock_accelerator, mock_logger)
        assert len(dataset) == len(mock_dataset["train"])  # Ensure the dataset is loaded correctly

    @it("should prepare the training dataset correctly")
    def test_prepare_train_dataset(self, mock_args, mock_dataset, mock_accelerator):
        dataset = prepare_train_dataset(mock_args, mock_dataset["train"], mock_accelerator)
        assert "pixel_values" in dataset[0]
        assert "conditioning_pixel_values" in dataset[0]

    @it("should collate the dataset correctly")
    def test_collate_fn(self):
        mock_data = {
            "pixel_values": torch.tensor(np.zeros((3, 256, 256), dtype=np.float32)),
            "conditioning_pixel_values": torch.tensor(np.zeros((3, 256, 256), dtype=np.float32)),
            "prompt_embeds": torch.tensor([1.0]),
            "text_embeds": torch.tensor([1.0]),
            "time_ids": torch.tensor([1.0]),
            "syn_or_real": torch.tensor([1.0]),
        }
        batch = [mock_data, mock_data]

        collated_batch = collate_fn(batch)

        assert isinstance(collated_batch, dict)
        assert "pixel_values" in collated_batch
        assert "conditioning_pixel_values" in collated_batch
        assert "prompt_ids" in collated_batch
        assert "unet_added_conditions" in collated_batch
        assert "text_embeds" in collated_batch["unet_added_conditions"]
        assert "time_ids" in collated_batch["unet_added_conditions"]
        assert "syn_or_real" in collated_batch

        assert collated_batch["pixel_values"].shape[0] == 2
        assert collated_batch["conditioning_pixel_values"].shape[0] == 2
        assert collated_batch["prompt_ids"].shape[0] == 2
        assert collated_batch["unet_added_conditions"]["text_embeds"].shape[0] == 2
        assert collated_batch["unet_added_conditions"]["time_ids"].shape[0] == 2
        assert collated_batch["syn_or_real"].shape[0] == 2