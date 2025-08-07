import pytest
from unittest import mock
import torch
from controlnet_sdxl.train_controlnet_sdxl import (
    main,
)  # Adjust the import based on where your main function is located


@pytest.mark.describe("Main Function Tests")
class TestMainFunction:

    @pytest.mark.it(
        "should raise ValueError if report_to is wandb and hub_token is provided"
    )
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.logging")
    def test_report_to_wandb_with_hub_token(self, mock_logging):
        args = mock.Mock()
        args.report_to = "wandb"
        args.hub_token = "some_token"

        with pytest.raises(ValueError):
            main(args)

    @pytest.mark.skip
    @pytest.mark.it("should raise ValueError if mixed_precision is bf16 on MPS")
    @mock.patch("torch.backends.mps.is_available", return_value=True)
    def test_mixed_precision_bf16_on_mps(self, mock_mps):
        args = mock.Mock()
        args.report_to = None
        args.hub_token = None
        args.output_dir = "output"
        args.logging_dir = "logs"
        args.mixed_precision = "bf16"  # Set to a valid string
    

        with pytest.raises(ValueError):
            main(args)

    @pytest.mark.skip
    @pytest.mark.it("should create the output directory if it does not exist")
    @mock.patch("os.makedirs")
    def test_create_output_directory(self, mock_makedirs):
        args = mock.Mock()
        args.report_to = None
        args.hub_token = None
        args.output_dir = "output"
        args.logging_dir = "logs"
        args.push_to_hub = False
        args.pretrained_model_name_or_path = "some_model"
        args.mixed_precision = "fp16"
        args.log_type = "mlflow"
        args.gradient_accumulation_steps = 1
        args.seed = 404

        main(args)
        mock_makedirs.assert_called_once_with("output", exist_ok=True)

    @pytest.mark.skip
    @pytest.mark.it("should call create_repo if push_to_hub is True")
    @mock.patch("huggingface_hub.create_repo")
    def test_create_repo_called_if_push_to_hub(self, mock_create_repo):
        args = mock.Mock()
        args.output_dir = "output"
        args.logging_dir = "logs"
        args.push_to_hub = True
        args.hub_model_id = None
        args.hub_token = "some_token"
        args.pretrained_model_name_or_path = "some_model"
        args.mixed_precision = "fp16"
        args.log_type = "mlflow"
        args.revision = "idontknow"

        main(args)
        mock_create_repo.assert_called_once()

    @pytest.mark.skip
    @pytest.mark.it("should log information if accelerator is main process")
    @mock.patch("accelerate.Accelerator")
    @mock.patch("accelerate.Accelerator.unwrap_model")
    @mock.patch("logging.basicConfig")
    @mock.patch("transformers.models.auto.tokenization_auto.get_tokenizer_config")
    @mock.patch("transformers.models.auto.tokenization_auto.tokenizer_class_from_name")
    @mock.patch("transformers.PretrainedConfig.from_pretrained")
    @mock.patch("controlnet_sdxl.model.import_model_class_from_model_name_or_path")
    @mock.patch("diffusers.schedulers.scheduling_ddpm.DDPMScheduler.load_config")
    @mock.patch("diffusers.schedulers.scheduling_ddpm.DDPMScheduler.from_pretrained")
    @mock.patch("transformers.modeling_utils.PreTrainedModel.from_pretrained")
    @mock.patch("diffusers.models.modeling_utils.ModelMixin.from_pretrained")
    def test_logging_if_main_process(
        self,
        mock_modeling_modelmixin_pretrained,
        mock_modeling_pretrained,
        mock_from_pretrained_ddpms,
        mock_load_config,
        mock_controlnet_sdxl_model,
        mock_from_pretrained,
        mock_tokenizer_class,
        mock_tokenizer_config,
        mock_logging,
        mock_unwrap_model,
        mock_accelerator,
    ):
            
        # Create a mock controlnet model
        mock_controlnet = mock.Mock()
        mock_controlnet.dtype = torch.float32
        # Set the dtype to a valid tensor type
        #mock_controlnet.dtype.return_value = torch.float32  # Simulate proper tensor dtype

        # Simulate the behavior of the unwrap_model function
        #mock_unwrap_model = mock.Mock(return_value=mock_controlnet)
        # Ensure the from_pretrained mock returns this mock controlnet
        mock_modeling_modelmixin_pretrained.return_value = mock_controlnet

        args = mock.Mock()
        args.output_dir = "output"
        args.logging_dir = "logs"
        args.push_to_hub = False
        args.pretrained_model_name_or_path = "some_model"
        args.mixed_precision = "fp16"
        args.gradient_accumulation_steps = 2
        args.report_to = "mlflow"
        args.seed = 404
        args.pretrained_model_name_or_path = "dummy_path"
        args.enable_npu_flash_attention = False
        args.enable_xformers_memory_efficient_attention = False

        accelerator_instance = mock_accelerator.return_value
        accelerator_instance.is_local_main_process = True
        accelerator_instance.mixed_precision = (
            "fp16"  # Set to a valid mixed precision value
        )
        mock_from_pretrained.return_value.architectures = ["CLIPTextModel"]

        with mock.patch("accelerate.Accelerator", return_value=accelerator_instance):
            with mock.patch("logging.basicConfig") as mock_logging:
                main(args)
                mock_logging.assert_called_once()  # Check if logging was called
