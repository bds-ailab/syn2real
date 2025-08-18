import pytest
from unittest import mock
from PIL import Image
import numpy as np
from model_eval.train_segmenter import (
    preprocess_train,
    reduce_labels,
    augment,
    read_train_config,
)
from model_eval.labels import reduced_labels


@pytest.mark.describe("Train Segmenter Tests")
class TestTrainSegmenter:

    @pytest.mark.it("should read config.yml file correctly")
    def test_read_train_config(self, tmp_path):
        # Create temporary config file
        config_file = tmp_path / "config.yml"
        config_file.write_text("key: value")

        with mock.patch(
            "model_eval.train_segmenter.open", mock.mock_open(read_data="key: value")
        ), mock.patch(
            "model_eval.train_segmenter.yaml.safe_load", return_value={"key": "value"}
        ):
            config = read_train_config()
            assert isinstance(config, dict)
            assert config["key"] == "value"

    @pytest.mark.it("should preprocess the images correctly")
    def test_preprocess_train(self):
        img = Image.fromarray(np.uint8(np.random.rand(128, 128, 3) * 255)).convert(
            "RGB"
        )
        label = Image.fromarray(np.uint8(np.random.rand(128, 128) * 34)).convert("L")

        examples = {"pixel_values": [img], "label": [label]}
        processed = preprocess_train(examples)

        assert "pixel_values" in processed
        assert "label" in processed
        assert isinstance(processed["pixel_values"][0], Image.Image)
        assert isinstance(processed["label"][0], Image.Image)

    @pytest.mark.it("should reduce labels correctly")
    def test_reduce_labels(self):
        # Create label image with values between 0-33
        label_array = np.random.randint(0, 34, (64, 64), dtype=np.uint8)
        label_image = Image.fromarray(label_array)

        reduced = reduce_labels(label_image)

        assert isinstance(reduced, Image.Image)
        reduced_array = np.array(reduced)

        # All label values must be between 0 and len(reduced_labels) - 1
        assert reduced_array.min() >= 0
        assert reduced_array.max() <= len(reduced_labels) - 1

    @pytest.mark.it("should return augmented images and labels as PIL images")
    def test_augment_function_output_type(self):
        img = Image.fromarray(np.uint8(np.random.rand(512, 512, 3) * 255)).convert(
            "RGB"
        )
        label = Image.fromarray(np.uint8(np.random.randint(0, 34, (512, 512))))

        aug_img, aug_label = augment(img, label)

        assert isinstance(aug_img, Image.Image)
        assert isinstance(aug_label, Image.Image)

    @pytest.mark.it("should preserve input size after augmentation")
    def test_augment_preserves_shape(self):
        width, height = 128, 128
        img = Image.fromarray(np.uint8(np.random.rand(height, width, 3) * 255)).convert(
            "RGB"
        )
        label = Image.fromarray(np.uint8(np.random.randint(0, 34, (height, width))))

        aug_img, aug_label = augment(img, label)

        assert aug_img.size == (width, height)
        assert aug_label.size == (width, height)
