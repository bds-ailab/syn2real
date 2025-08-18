import pytest
from unittest import mock
import numpy as np
import torch
from PIL import Image
from controlnet_sdxl.log import log_validation, parse_args

it = pytest.mark.it
describe = pytest.mark.describe


@describe("Test validation logging and argument parsing functions")
class TestValidationFunctions:

    @it("should log validation images correctly")
    @mock.patch("controlnet_sdxl.log.StableDiffusionXLControlNetPipeline")
    @mock.patch("controlnet_sdxl.log.AutoencoderKL")
    @mock.patch("controlnet_sdxl.log.ControlNetModel")
    @mock.patch("controlnet_sdxl.log.gc.collect")
    @mock.patch("controlnet_sdxl.log.torch.cuda.empty_cache")
    @mock.patch("diffusers.UniPCMultistepScheduler.from_config")
    @mock.patch("torch.autocast")
    @mock.patch("PIL.Image.open")
    def test_log_validation(
        self,
        mock_open,
        mock_autocast,
        mock_unipc,
        mock_empty_cache,
        mock_gc,
        mock_ControlNetModel,
        mock_AutoencoderKL,
        mock_pipeline,
    ):
        # Mocking the pipeline's output
        class MockPipeline:
            def __call__(self, *args, **kwargs):
                return self

            @property
            def images(self):
                return [Image.fromarray(np.zeros((512, 1024, 3), dtype=np.uint8))]

        mock_pipeline.return_value = MockPipeline()

        # Mocking the models
        mock_ControlNetModel.from_pretrained.return_value = mock.Mock()
        mock_AutoencoderKL.from_pretrained.return_value = mock.Mock()

        # Mocking arguments
        class MockArgs:
            pretrained_model_name_or_path = "model/path"
            output_dir = "output/path"
            resolution = 512
            num_validation_images = 1
            validation_image = ["path/to/validation_image.png"]
            validation_prompt = ["A test prompt"]
            revision = None
            variant = None
            seed = None
            enable_xformers_memory_efficient_attention = True

        mock_args = MockArgs()
        mock_accelerator = mock.Mock()
        mock_tracker_tensorboard = mock.Mock()
        mock_tracker_tensorboard.name = "tensorboard"
        mock_tracker_other = mock.Mock()
        mock_tracker_other.name = "other"
        mock_accelerator.trackers = [
            mock_tracker_tensorboard,
            mock_tracker_other,
        ]
        mock_weight_dtype = torch.float32
        mock_logger = mock.Mock()

        # Mocking the validation image
        mock_open = mock.mock_open(read_data="an_image")
        with mock.patch("builtins.open", mock_open):
            image_logs = log_validation(
                vae=mock.Mock(),
                unet=mock.Mock(),
                controlnet=mock.Mock(),
                args=mock_args,
                accelerator=mock_accelerator,
                weight_dtype=mock_weight_dtype,
                step=1,
                logger=mock_logger,
                is_final_validation=False,
            )

        assert len(image_logs) == 1  # Ensure one log entry is created

    @it("should parse command line arguments correctly")
    def test_parse_args(self):
        args = [
            "--pretrained_model_name_or_path",
            "model/path",
            "--output_dir",
            "output/path",
            "--validation_image",
            "path/to/image.png",
            "--validation_prompt",
            "A test prompt",
            "--dataset_name",
            "A_dataset_name",
        ]
        parsed_args = parse_args(args)

        assert parsed_args.pretrained_model_name_or_path == "model/path"
        assert parsed_args.output_dir == "output/path"
        assert parsed_args.validation_image == ["path/to/image.png"]
        assert parsed_args.validation_prompt == ["A test prompt"]

    @it("should raise ValueError if both dataset_name and train_data_dir are specified")
    def test_parse_args_error(self):
        args = [
            "--pretrained_model_name_or_path",
            "model/path",
            "--output_dir",
            "output/path",
            "--dataset_name",
            "dataset_name",
            "--train_data_dir",
            "train_data_dir",
        ]
        with pytest.raises(
            ValueError,
            match="Specify only one of `--dataset_name` or `--train_data_dir`",
        ):
            parse_args(args)
