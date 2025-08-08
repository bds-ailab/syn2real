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

    @pytest.mark.it("should create the output directory if it does not exist")
    @mock.patch("transformers.AutoTokenizer.from_pretrained")
    @mock.patch("os.makedirs")
    @mock.patch(
        "controlnet_sdxl.train_controlnet_sdxl.import_model_class_from_model_name_or_path"
    )
    @mock.patch("transformers.PretrainedConfig.from_pretrained")
    @mock.patch("diffusers.DDPMScheduler.from_pretrained")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.AutoencoderKL.from_pretrained")
    @mock.patch(
        "controlnet_sdxl.train_controlnet_sdxl.UNet2DConditionModel.from_pretrained"
    )
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.ControlNetModel.from_pretrained")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.unwrap_model")
    @mock.patch(
        "controlnet_sdxl.train_controlnet_sdxl.Accelerator.num_processes",
        return_value=2,
    )
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.Accelerator.load_state")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.Accelerator.prepare")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.Accelerator.clip_grad_norm_")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.Accelerator.backward")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.Accelerator.save_state")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.AdamW8bit")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.get_train_dataset")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.Hasher.hash")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.torch.utils.data.DataLoader")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.torch.randn_like")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.torch.randint")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.next")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.get_scheduler")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.torch.nn.functional.mse_loss")
    @mock.patch("controlnet_sdxl.train_controlnet_sdxl.log_validation")
    def test_create_output_directory(
        self,
        mock_log_valid,
        mock_mse_loss,
        mock_get_scheduler,
        mock_next,
        mock_randint,
        mock_randn,
        mock_dataloader,
        mock_hash,
        mock_get_train_dataset,
        mock_adamw,
        mock_accelerator_save_state,
        mock_accelerator_backward,
        mock_accelerator_clip_grad_norm_,
        mock_accelerator_prepare,
        mock_accelerator_lod_state,
        mock_num_processes,
        mock_unwrap_model,
        mock_controlnet_pretrained,
        mock_unet_pretrained,
        mock_kl_pretrained,
        mock_ddpms,
        mock_pretrained_config,
        mock_import,
        mock_makedirs,
        mock_tokenizer_from_pretrained,
    ):
        mock_controlnet = mock.Mock()
        mock_controlnet.dtype = torch.float32
        mock_controlnet_pretrained.return_value = mock_controlnet
        mock_unwrap_model.return_value = mock_controlnet

        mock_unet_pretrained = mock.Mock()
        mock_output_tensor = mock.Mock()
        mock_unet_pretrained.return_value = torch.randn(1, 4, 64, 64)

        mock_import.return_value = mock.Mock()
        mock_tokenizer_from_pretrained.return_value = mock.Mock()

        mock_pixel_values = mock.Mock()
        mock_pixel_values.to.return_value = mock_pixel_values  # simulate .to()
        mock_randn.return_value = torch.randn(1, 4, 64, 64)
        mock_batch = {
            "pixel_values": mock_pixel_values,
            "conditioning_pixel_values": mock_pixel_values,
            "prompt_ids": 0,
            "unet_added_conditions": "yes",
        }

        mock_controlnet_accelerate = mock.Mock()
        mock_down_samples = [mock.Mock(), mock.Mock(), mock.Mock()]
        mock_mid_sample = mock.Mock()

        for s in mock_down_samples:
            s.to.return_value = s
        mock_mid_sample.to.return_value = mock_mid_sample
        mock_controlnet_accelerate.return_value = (mock_down_samples, mock_mid_sample)
        mock_lr = mock.Mock()
        mock_last_lr = [1, 2]
        mock_lr.get_last_lr.return_value = mock_last_lr
        mock_accelerator_prepare.return_value = (
            mock_controlnet_accelerate,
            mock.Mock(),
            [mock_batch],
            mock_lr,
        )
        # This sets up what .architectures[0] will return
        mock_config_instance = mock.Mock()
        mock_config_instance.architectures = ["CLIPTextModel"]
        mock_pretrained_config.return_value = mock_config_instance

        mock_scheduler = mock.Mock()
        mock_scheduler.config.prediction_type = "epsilon"
        mock_scheduler.add_noise.return_value = mock.Mock()
        mock_ddpms.return_value = mock_scheduler

        args = mock.Mock()
        args.revision = "main"
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
        args.enable_npu_flash_attention = False
        args.enable_xformers_memory_efficient_attention = False
        args.learning_rate = 1
        args.scale_lr = 1
        args.train_batch_size = 1
        args.dataloader_num_workers = 2
        args.max_train_steps = 1
        args.validation_prompt = "dummy_prompt"
        args.validation_image = "dummy_image"
        args.resume_from_checkpoint = "latest"
        args.checkpointing_steps = 1
        args.checkpoints_total_limit = 2
        args.validation_steps = 2

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
        # mock_controlnet.dtype.return_value = torch.float32  # Simulate proper tensor dtype

        # Simulate the behavior of the unwrap_model function
        # mock_unwrap_model = mock.Mock(return_value=mock_controlnet)
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
