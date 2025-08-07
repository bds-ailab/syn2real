import pytest
import numpy as np
import torch
import cv2
from unittest import mock
from PIL import Image
from controlnet_sdxl.inference import main, sharpen, correct

it = pytest.mark.it
describe = pytest.mark.describe

@describe("Test image generation and processing functions")
class TestInference:

    @it("should generate images from conditioning images and prompts")
    @mock.patch("controlnet_sdxl.inference.StableDiffusionXLControlNetPipeline")
    @mock.patch("controlnet_sdxl.inference.load_image")
    def test_main(self, mock_load_image, mock_pipeline):
        # Mocking the pipeline's output
        class MockPipeline:
            def __call__(self, *args, **kwargs):
                return self

            @property
            def images(self):
                return [Image.fromarray(np.zeros((512, 1024, 3), dtype=np.uint8))]

        mock_pipeline.return_value = MockPipeline()

        cond_image_paths = ["path/to/cond_image1.png", "path/to/cond_image2.png"]
        prompts = ["A test prompt"]
        seed = 42

        # Mocking load_image to return a dummy image
        mock_load_image.side_effect = lambda path: Image.fromarray(np.zeros((512, 1024, 3), dtype=np.uint8))

        images = main(mock_pipeline, cond_image_paths, prompts, seed)

        assert len(images) == 1  # Ensure one image is generated
        assert isinstance(images[0], Image.Image)  # Ensure the output is an image

    @it("should sharpen the image correctly")
    def test_sharpen(self):
        # Create a dummy image
        dummy_image = np.zeros((512, 1024, 3), dtype=np.uint8)
        dummy_image_path = "dummy_image.png"
        sharpened_image_path = "sharpened_image.png"

        # Save the dummy image
        cv2.imwrite(dummy_image_path, dummy_image)

        # Call the sharpen function
        sharpen(dummy_image_path, sharpened_image_path)

        # Load the sharpened image to verify
        sharpened_image = cv2.imread(sharpened_image_path)

        assert sharpened_image is not None  # Ensure the image was saved
        assert sharpened_image.shape == dummy_image.shape  # Ensure the shape is the same

    @it("should correct brightness and contrast of the image")
    def test_correct(self):
        # Create a dummy image
        dummy_image = np.zeros((512, 1024, 3), dtype=np.uint8)

        # Test brightness and contrast correction
        corrected_image = correct(dummy_image, brightness=300, contrast=150)

        assert corrected_image is not None  # Ensure the image is returned
        assert corrected_image.shape == dummy_image.shape  # Ensure the shape is the same