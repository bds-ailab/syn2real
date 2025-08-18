import os
import types
import torch
import pytest
from unittest import mock

it = pytest.mark.it
describe = pytest.mark.describe


def make_args(tmp_path):
    """Build a minimal args namespace for one quick training iteration."""
    args = types.SimpleNamespace(
        # sorties / dossier
        output_dir=str(tmp_path / "out"),
        dataset_folder=str(tmp_path / "ds"),
        # acceleration / log
        gradient_accumulation_steps=1,
        report_to=None,
        seed=42,
        revision="main",
        # LoRA / ranks
        lora_rank_unet=4,
        lora_rank_vae=4,
        # options UNet
        enable_xformers_memory_efficient_attention=False,
        gradient_checkpointing=False,
        allow_tf32=False,
        # training
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_weight_decay=0.0,
        adam_epsilon=1e-8,
        lr_scheduler="constant",
        lr_warmup_steps=0,
        max_train_steps=1,
        max_train_epochs=1,
        lr_num_cycles=1,
        lr_power=1.0,
        dataloader_num_workers=0,
        train_batch_size=2,
        train_img_prep="resize_256",
        val_img_prep="resize_256",
        tracker_project_name="test",
        # GAN
        gan_disc_type="vagan_clip",
        gan_loss_type="hinge",
        viz_freq=999999,
        checkpointing_steps=999999,
        lambda_cycle=1.0,
        lambda_cycle_lpips=0.0,
        lambda_gan=1.0,
        lambda_idt=0.5,
        lambda_idt_lpips=0.0,
        max_grad_norm=1.0,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.dataset_folder, exist_ok=True)
    return args


@describe("Training main() minimal smoke test")
class TestTrainScript:
    @it("runs one tiny step without touching real GPU / network / files")
    @mock.patch("gan.train_cyclegan.tqdm")  # progress bar
    @mock.patch("gan.train_cyclegan.wandb", create=True)  # in case report_to='wandb'
    @mock.patch("gan.train_cyclegan.torch.save")  # avoid writing checkpoints
    @mock.patch("gan.train_cyclegan.get_peft_model_state_dict")  # avoid peft internals
    @mock.patch("gan.train_cyclegan.get_folder_features")
    @mock.patch("gan.train_cyclegan.build_feature_extractor")
    @mock.patch("gan.train_cyclegan.get_scheduler")
    @mock.patch("gan.train_cyclegan.bnb.optim.AdamW8bit")
    @mock.patch("gan.train_cyclegan.UnpairedDataset")
    @mock.patch("gan.train_cyclegan.build_transform")
    @mock.patch("gan.train_cyclegan.CycleGAN_Turbo")
    @mock.patch("gan.train_cyclegan.VAE_encode")
    @mock.patch("gan.train_cyclegan.VAE_decode")
    @mock.patch("gan.train_cyclegan.initialize_unet")
    @mock.patch("gan.train_cyclegan.initialize_vae")
    @mock.patch("gan.train_cyclegan.make_1step_sched")
    @mock.patch("gan.train_cyclegan.CLIPTextModel")
    @mock.patch("gan.train_cyclegan.AutoTokenizer")
    @mock.patch("gan.train_cyclegan.set_seed")
    @mock.patch("gan.train_cyclegan.Accelerator")
    @mock.patch("torch.Tensor.cuda", new=lambda self: self)  # no CUDA init from tensors
    @mock.patch("torch.cuda.empty_cache")
    def test_main_one_step(
        self,
        mock_empty_cache,
        mock_accelerator_cls,
        mock_set_seed,
        mock_tokenizer_cls,
        mock_clip_cls,
        mock_make_sched,
        mock_init_vae,
        mock_init_unet,
        mock_vae_decode_cls,
        mock_vae_encode_cls,
        mock_cyclegan_cls,
        mock_build_transform,
        mock_unpaired_dataset_cls,
        mock_adam8bit_cls,
        mock_get_scheduler,
        mock_build_feat_extractor,
        mock_get_folder_features,
        mock_get_peft_sd,
        mock_torch_save,
        mock_wandb,
        mock_tqdm,
        tmp_path,
    ):
        progress = mock.Mock()
        progress.update = mock.Mock()
        progress.set_postfix = mock.Mock()

        def tqdm_side_effect(*args, **kwargs):
            # Si des kwargs sont fournis (initial/desc/disable...), on est dans le cas "barre de progression"
            if kwargs:
                return progress
            # Sinon, on est dans le cas "tqdm(iterable)" -> on renvoie l'itérable lui-même
            # pour que "for x in tqdm(iterable)" fonctionne.
            if args:
                return args[0]
            return progress  # fallback

        mock_tqdm.side_effect = tqdm_side_effect

        # ---------------------------
        # Arrange: minimal args & fs
        # ---------------------------
        args = make_args(tmp_path)
        # create dataset folders structure
        os.makedirs(os.path.join(args.dataset_folder, "test_A"), exist_ok=True)
        os.makedirs(os.path.join(args.dataset_folder, "test_B"), exist_ok=True)

        # ---------------------------
        # Mock Accelerator
        # ---------------------------
        acc = mock.Mock()
        acc.device = "cpu"
        acc.num_processes = 1
        acc.is_main_process = True
        acc.is_local_main_process = True
        acc.sync_gradients = True
        acc.prepare.side_effect = lambda *xs: xs  # identity
        acc.clip_grad_norm_ = mock.Mock()
        acc.init_trackers = mock.Mock()
        acc.unwrap_model.side_effect = lambda x: x
        acc.log = mock.Mock()
        acc.trackers = []  # no wandb tracker branch

        class _DummyCtx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False  # ne supprime pas les exceptions

        acc.accumulate.return_value = _DummyCtx()
        acc.backward = mock.Mock()
        mock_accelerator_cls.return_value = acc

        # ---------------------------
        # Tokenizer & Text Encoder
        # ---------------------------

        # Tokenizer mock
        tok = mock.MagicMock()
        tok.model_max_length = 77

        # When tokenizer(...) is called, we need an object with .input_ids tensor
        class Tokens:
            def __init__(self):
                self.input_ids = torch.randint(0, 10, (1, 77))

        tok.return_value = Tokens()  # tokenizer(...) -> Tokens()
        mock_tokenizer_cls.from_pretrained.return_value = tok

        # Text encoder mock (callable)
        text_enc = mock.MagicMock()
        text_enc.cuda.return_value = text_enc
        # text_encoder(...) should return a tuple where [0] is the embeddings tensor
        text_enc.return_value = (torch.randn(1, 77, 768),)
        text_enc.to.return_value = text_enc  # allow chaining .to(...)
        text_enc.requires_grad_.return_value = None
        mock_clip_cls.from_pretrained.return_value = text_enc

        # ---------------------------
        # make_1step_sched
        # ---------------------------
        sched = mock.Mock()
        sched.config = types.SimpleNamespace(num_train_timesteps=1000)
        mock_make_sched.return_value = sched

        # ---------------------------
        # initialize_unet / vae
        # ---------------------------
        # minimal UNet-like with conv_in and required methods
        class DummyUnet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_in = torch.nn.Conv2d(3, 3, 1)

            def to(self, *a, **k):
                return self

            def enable_xformers_memory_efficient_attention(self):
                pass

            def enable_gradient_checkpointing(self):
                pass

            def named_modules(self):
                return []

            def parameters(self):
                return []

        unet = DummyUnet()
        l_encoder = ["e1"]
        l_decoder = ["d1"]
        l_other = ["o1"]
        mock_init_unet.return_value = (unet, l_encoder, l_decoder, l_other)

        class DummyVAE(torch.nn.Module):
            def to(self, *a, **k):
                return self

            def state_dict(self):
                return {"w": torch.tensor(1.0)}

            def parameters(self):
                return []

        vae_a2b = DummyVAE()
        vae_lora_modules = ["l1", "l2"]
        mock_init_vae.return_value = (vae_a2b, vae_lora_modules)

        # VAE encode/decode wrappers
        vae_enc = mock.Mock()
        vae_dec = mock.Mock()
        mock_vae_encode_cls.return_value = vae_enc
        mock_vae_decode_cls.return_value = vae_dec

        # ---------------------------
        # Discriminators (vision_aided_loss)
        # ---------------------------
        with mock.patch(
            "gan.train_cyclegan.vision_aided_loss.Discriminator"
        ) as disc_cls:
            # Use MagicMock so calling the instance works: disc(...) -> return_value
            disc = mock.MagicMock()
            disc.return_value = torch.zeros(
                2, 1
            )  # net_disc(fake, for_G=...) returns batch
            disc.parameters.return_value = []
            # make .cv_ensemble.requires_grad_(False) legal
            disc.cv_ensemble = mock.MagicMock()
            disc.cv_ensemble.requires_grad_.return_value = None
            # iterate named_modules() without 'attn' entries
            disc.named_modules.return_value = []
            disc_cls.return_value = disc

            # ---------------------------
            # Optimizers & Schedulers
            # ---------------------------
            opt_gen = mock.Mock()
            opt_disc = mock.Mock()

            for opt in (opt_gen, opt_disc):
                opt.step = mock.Mock()
                opt.zero_grad = mock.Mock()

            mock_adam8bit_cls.side_effect = [opt_gen, opt_disc]

            sch_gen = mock.Mock()
            sch_disc = mock.Mock()
            sch_gen.step = mock.Mock()
            sch_disc.step = mock.Mock()
            mock_get_scheduler.side_effect = [sch_gen, sch_disc]

            # ---------------------------
            # Dataset & DataLoader
            # ---------------------------
            dataset = mock.Mock()
            # attributes used in code:
            dataset.fixed_caption_src = "src caption"
            dataset.fixed_caption_tgt = "tgt caption"
            mock_unpaired_dataset_cls.return_value = dataset

            # simple dataloader yielding a single batch
            batch = {
                "pixel_values_src": torch.randn(2, 3, 8, 8),
                "pixel_values_tgt": torch.randn(2, 3, 8, 8),
            }
            dataloader = [batch]
            with mock.patch("torch.utils.data.DataLoader", return_value=dataloader):

                # ---------------------------
                # Val transform used for FID
                # ---------------------------
                # Should return PIL image with .save()
                mock_build_transform.return_value = lambda im: im

                # ---------------------------
                # CleanFID bits
                # ---------------------------
                mock_build_feat_extractor.return_value = mock.Mock()
                # return features array (N,d) to make mean/cov work
                mock_get_folder_features.return_value = torch.randn(4, 64).numpy()

                # ---------------------------
                # LPIPS
                # ---------------------------
                with mock.patch("gan.train_cyclegan.lpips.LPIPS") as lpips_cls:
                    lp = mock.MagicMock()  # MagicMock, pas Mock
                    lp.cuda.return_value = None
                    lp.requires_grad_.return_value = None
                    lp.return_value = torch.zeros(
                        2, 1, 1, 1
                    )  # lp(img1, img2) -> batch tensor
                    lpips_cls.return_value = lp

                    # ---------------------------
                    # CycleGAN forward_with_networks
                    # ---------------------------
                    mock_cyclegan_cls.get_traininable_params = staticmethod(
                        lambda *args, **kwargs: []
                    )

                    # Return a tensor with same shape as input image
                    def fwd_like(x, *a, **k):
                        return x

                    mock_cyclegan_cls.forward_with_networks = staticmethod(fwd_like)

                    # ---------------------------
                    # Act: call main(args)
                    # ---------------------------
                    from gan.train_cyclegan import main

                    main(args)

        # ---------------------------
        # Assert: basic calls occurred
        # ---------------------------
        mock_set_seed.assert_called_once_with(42)
        mock_tokenizer_cls.from_pretrained.assert_called_once()
        mock_clip_cls.from_pretrained.assert_called_once()
        mock_make_sched.assert_called_once()
        mock_init_unet.assert_called_once()
        mock_init_vae.assert_called_once()
        mock_unpaired_dataset_cls.assert_called_once()
        mock_build_transform.assert_called_once_with("resize_256")
        mock_get_scheduler.assert_any_call(
            "constant",
            optimizer=mock.ANY,
            num_warmup_steps=0,
            num_training_steps=1,
            num_cycles=1,
            power=1.0,
        )
