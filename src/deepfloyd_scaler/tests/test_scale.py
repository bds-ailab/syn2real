import pytest
from unittest import mock
import torch
from transformers import T5EncoderModel
from diffusers import DiffusionPipeline
from PIL import Image
import os
import json
from deepfloyd_scaler.scale import *


@pytest.mark.describe("Test Image Processing Functions")
class TestImageProcessing:

    @pytest.mark.it("should flush GPU memory correctly")
    def test_flush(self):
        # This function does not return anything, so we simply call it to ensure it runs without errors
        flush()
        assert True  # Just to ensure the test runs

    @pytest.mark.it("should encode prompts correctly")
    @mock.patch("transformers.T5EncoderModel.from_pretrained")
    @mock.patch("diffusers.DiffusionPipeline.from_pretrained")
    def test_encode_prompts(self, mock_diffusion_pipeline, mock_t5_encoder):
        # Setup mock return values
        mock_t5_encoder.return_value = mock.Mock()
        mock_diffusion_pipeline.return_value = mock.Mock()
        mock_diffusion_pipeline().encode_prompt.return_value = (
            torch.tensor([1.0]),
            torch.tensor([0.0]),
        )

        prompts = ["Hello world", "Test prompt"]
        embeddings = encode_prompts(prompts)

        assert len(embeddings) == 2
        assert torch.equal(embeddings[0][0], torch.tensor([1.0]))
        assert torch.equal(embeddings[0][1], torch.tensor([0.0]))
        mock_t5_encoder.assert_called_once()
        mock_diffusion_pipeline.assert_called()

    @pytest.mark.it("should transform image correctly")
    @mock.patch("PIL.Image.open")
    @mock.patch("torchvision.transforms.Compose")
    def test_transform_img(self, mock_transforms_compose, mock_image_open):
        # Mock the image and transformations
        mock_image = mock.Mock()
        mock_image.size = (200, 200)
        mock_image_open.return_value = mock_image

        mock_transform = mock.Mock()
        mock_transform.return_value = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]]]
        )  # Mock tensor output with 3 dimensions
        mock_transforms_compose.return_value = mock_transform

        img_tensor = transform_img("fake_path.jpg", size=100)

        assert img_tensor.shape == (
            1,
            1,
            2,
            2,
        )  # Check the shape of the returned tensor
        mock_image_open.assert_called_once_with("fake_path.jpg")
        mock_transform.assert_called_once()

    @pytest.mark.it("should process images and prompts correctly")
    @mock.patch("transformers.T5EncoderModel.from_pretrained")
    @mock.patch("diffusers.DiffusionPipeline.from_pretrained")
    @mock.patch("deepfloyd_scaler.scale.transform_img")
    @mock.patch("os.listdir")
    @mock.patch("PIL.Image.Image.save")
    @mock.patch("PIL.Image.Image")
    def test_process(
        self,
        mock_image,
        mock_image_save,
        mock_listdir,
        mock_transform_img,
        mock_diffusion_pipeline,
        mock_t5_encoder,
    ):
        # Configure the mock for T5 encoder
        mock_t5_encoder.return_value = mock.Mock()

        # Configure the mock for the diffusion pipelines
        mock_pipe1 = mock.Mock()
        mock_pipe2 = mock.Mock()

        # Set up the return values for the diffusion pipelines
        mock_diffusion_pipeline.side_effect = [mock_pipe1, mock_pipe1, mock_pipe2]

        # Mock the output of the first pipeline to return a tensor
        mock_pipe1.return_value.images = torch.tensor(
            [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
        )  # Mock tensor output
        mock_pipe2.return_value.images = [
            mock.Mock(spec=Image.Image)
        ]  # Mock the output as a list with 1 PIL Image

        # Mock the encode_prompt method to return tensors
        mock_pipe1.encode_prompt.return_value = (
            torch.tensor([1.0]),  # mock prompt embeddings
            torch.tensor([0.0]),  # mock negative prompt embeddings
        )
        # Set up the list directory mock
        mock_listdir.return_value = ["out1_image1.png", "out1_image2.png"]

        # Define test inputs
        images_list = ["path/to/image1.png", "path/to/image2.png"]
        prompts_list = ["Prompt 1", "Prompt 2"]
        output_dir = "output/"

        # Call the process function
        process(images_list, prompts_list, size=100, output_dir=output_dir)

        # Ensure the mocks were called as expected
        mock_transform_img.assert_called()
        mock_t5_encoder.assert_called_once()
        mock_diffusion_pipeline.assert_called()
