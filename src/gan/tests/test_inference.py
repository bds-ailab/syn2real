import pytest
from unittest import mock
import os
import tempfile
import shutil
from PIL import Image
import torch
from gan.inference import main
from gan.training_utils import build_transform  # replace with actual module
from gan.cyclegan_turbo import CycleGAN_Turbo  # make sure import is correct

it = pytest.mark.it
describe = pytest.mark.describe


@describe("Test the inference script")
class TestInference:

    @it("Must run the whole inference pipeline")
    @mock.patch("gan.inference.CycleGAN_Turbo")
    @mock.patch("gan.inference.build_transform")
    @mock.patch("PIL.Image.open")
    @mock.patch("torchvision.transforms.ToTensor")
    @mock.patch("torchvision.transforms.ToPILImage")
    @mock.patch("torchvision.transforms.Normalize")
    @mock.patch("torch.Tensor.cuda", return_value=torch.randn(1, 3, 256, 256))
    def test_script_translation_pipeline(
        self,
        mock_cuda_tensor,
        mock_normalize,
        mock_to_pil,
        mock_to_tensor,
        mock_image_open,
        mock_build_transform,
        mock_cyclegan,
        monkeypatch,
    ):
        # Set up temporary directory for inputs and outputs
        temp_input_dir = tempfile.mkdtemp()
        temp_output_dir = tempfile.mkdtemp()

        try:
            # Create a fake input image file
            img_path = os.path.join(temp_input_dir, "test.jpg")
            Image.new("RGB", (256, 256)).save(img_path)

            # Mock arguments
            monkeypatch.setattr(
                "sys.argv",
                [
                    "script_name",
                    "--input_image",
                    temp_input_dir,
                    "--output_dir",
                    temp_output_dir,
                    "--model_name",
                    "pretrained_model_name",
                ],
            )

            # Mock behavior of external components
            mock_model = mock.Mock()
            mock_output_tensor = torch.randn(3, 256, 256)
            mock_model.return_value = [mock_output_tensor]
            mock_cyclegan.return_value = mock_model

            fake_transform = mock.Mock()
            mock_build_transform.return_value = fake_transform

            mock_img = Image.new("RGB", (256, 256))
            mock_image_open.return_value.convert.return_value = mock_img

            mock_to_tensor.return_value = mock.Mock(
                return_value=torch.randn(3, 256, 256)
            )
            mock_normalize.return_value = mock.Mock(
                return_value=torch.randn(3, 256, 256)
            )
            mock_to_pil.return_value = mock.Mock(return_value=mock_img)

            # Run main logic (under test)
            with mock.patch("torch.no_grad"):
                main()

            # Verify expected output file is created
            output_file = os.path.join(temp_output_dir, "test.jpg")
            assert os.path.exists(output_file)

        finally:
            shutil.rmtree(temp_input_dir)
            shutil.rmtree(temp_output_dir)
