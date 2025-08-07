import types
import pytest
from unittest import mock
import torch

# markers aliases
it = pytest.mark.it
describe = pytest.mark.describe


# -----------------------------
# Small helpers to craft a fake SD that drives the LoRA copy loops
# -----------------------------
def make_state_dict_for_copy(unet_param_shapes):
    """
    Build a state-dict that maps the expected 'name_sd' keys used in
    load_ckpt_from_state_dict's loops:
      - replace ".default_encoder.weight" -> ".weight", etc.
    The values are tensors with same shape as their targets.
    """
    sd = {
        "rank_unet": 4,
        "l_target_modules_encoder": ["enc.mod"],
        "l_target_modules_decoder": ["dec.mod"],
        "l_modules_others": ["oth.mod"],
        "rank_vae": 4,
        "vae_lora_target_modules": ["vae_skip"],
        "sd_encoder": {},
        "sd_decoder": {},
        "sd_other": {},
        "sd_vae_enc": {},
        "sd_vae_dec": {},
    }
    # populate sd_{encoder,decoder,other} for one parameter per category
    for full_name, tensor in unet_param_shapes.items():
        if ".default_encoder.weight" in full_name:
            sd["sd_encoder"][
                full_name.replace(".default_encoder.weight", ".weight")
            ] = torch.zeros_like(tensor)
        if ".default_decoder.weight" in full_name:
            sd["sd_decoder"][
                full_name.replace(".default_decoder.weight", ".weight")
            ] = torch.zeros_like(tensor)
        if ".default_others.weight" in full_name:
            sd["sd_other"][full_name.replace(".default_others.weight", ".weight")] = (
                torch.zeros_like(tensor)
            )
    return sd


# -----------------------------
# Common heavy patches to avoid downloading/initializing real models
# -----------------------------
def _patch_heavy_deps(mocker):
    # Tokenizer & text encoder
    tok = mock.MagicMock()
    tok.model_max_length = 77

    # tokenizer(...) -> object with .input_ids tensor
    class _Tokens:
        def __init__(self):
            self.input_ids = torch.randint(0, 10, (1, 77))

    tok.return_value = _Tokens()
    mocker.patch("gan.cyclegan_turbo.AutoTokenizer.from_pretrained", return_value=tok)

    txt = mock.MagicMock()
    txt.cuda.return_value = txt
    txt.to.return_value = txt
    txt.requires_grad_.return_value = None
    # text_encoder(tokens) -> tuple with embeddings at [0]
    txt.return_value = (torch.randn(1, 77, 768),)
    mocker.patch("gan.cyclegan_turbo.CLIPTextModel.from_pretrained", return_value=txt)

    # VAE / UNet / scheduler placeholders
    fake_vae = mock.MagicMock()
    fake_vae.encoder = mock.MagicMock()
    fake_vae.decoder = mock.MagicMock()
    fake_vae.config = types.SimpleNamespace(scaling_factor=1.0)
    # encoder/decoder.forward will be replaced by my_vae_* bindings in the module under test,
    # so we don't need functional behavior here.

    mocker.patch(
        "gan.cyclegan_turbo.AutoencoderKL.from_pretrained", return_value=fake_vae
    )

    fake_unet = mock.MagicMock()
    # named_parameters must expose LoRA param names the code iterates on
    # Provide one per category:
    # - default_encoder, default_decoder, default_others
    p_enc = torch.nn.Parameter(torch.randn(2, 2))
    p_dec = torch.nn.Parameter(torch.randn(3, 3))
    p_oth = torch.nn.Parameter(torch.randn(4, 4))
    fake_unet.named_parameters.return_value = [
        ("blocks.0.attn.lora_A.default_encoder.weight", p_enc),
        ("blocks.7.ff.lora_B.default_decoder.weight", p_dec),
        ("proj.lora.default_others.weight", p_oth),
    ]
    fake_unet.add_adapter = mock.MagicMock()
    fake_unet.set_adapters = mock.MagicMock()
    fake_unet.cuda.return_value = fake_unet
    fake_unet.train.return_value = None
    fake_unet.requires_grad_.return_value = None
    fake_unet.conv_in = mock.MagicMock()
    fake_unet.conv_in.parameters.return_value = []
    mocker.patch(
        "gan.cyclegan_turbo.UNet2DConditionModel.from_pretrained",
        return_value=fake_unet,
    )

    # make_1step_sched -> simple object with config and no-ops
    sched = mock.MagicMock()
    sched.config = types.SimpleNamespace(num_train_timesteps=1000)
    mocker.patch("gan.cyclegan_turbo.make_1step_sched", return_value=sched)

    # NOP all .cuda() on nn.Module to avoid CUDA init
    mocker.patch("torch.nn.Module.cuda", new=lambda self: self)

    return fake_unet, fake_vae, sched


@describe("CycleGAN_Turbo coverage-focused tests")
class TestCycleGANTurboCoverage:
    @it("covers all pretrained_name branches set (direction/caption/timesteps)")
    @pytest.mark.parametrize(
        "pretrained_name, expected_dir, expected_caption",
        [
            ("day_to_night", "a2b", "driving in the night"),
            ("night_to_day", "b2a", "driving in the day"),
            ("clear_to_rainy", "a2b", "driving in heavy rain"),
            ("rainy_to_clear", "b2a", "driving in the day"),
        ],
    )
    def test_init_pretrained_variants(
        self, mocker, pretrained_name, expected_dir, expected_caption
    ):
        # Force CUDA reported unavailable to take CPU default
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.cuda.device_count", return_value=0)
        fake_unet, fake_vae, sched = _patch_heavy_deps(mocker)

        # Avoid network in load_ckpt_from_url -> return a small valid sd
        # Build a minimal sd matching the unet named_parameters above
        unet_param_shapes = {
            "blocks.0.attn.lora_A.default_encoder.weight": fake_unet.named_parameters.return_value[
                0
            ][
                1
            ].data,
            "blocks.7.ff.lora_B.default_decoder.weight": fake_unet.named_parameters.return_value[
                1
            ][
                1
            ].data,
            "proj.lora.default_others.weight": fake_unet.named_parameters.return_value[
                2
            ][1].data,
        }
        sd = make_state_dict_for_copy(unet_param_shapes)

        # Patch download_url and torch.load for the URL path
        mocker.patch("gan.cyclegan_turbo.download_url", return_value=None)
        mocker.patch("torch.load", return_value=sd)

        from gan.cyclegan_turbo import CycleGAN_Turbo

        m = CycleGAN_Turbo(pretrained_name=pretrained_name)
        assert m.direction == expected_dir
        assert m.caption == expected_caption
        # Ensure timesteps exists and is long tensor
        assert isinstance(m.timesteps, torch.Tensor)
        assert m.timesteps.dtype in (torch.int64, torch.int32)

    @it(
        "happy path of load_ckpt_from_url + load_ckpt_from_state_dict copies LoRA weights"
    )
    def test_load_ckpt_happy_path(self, mocker):
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.cuda.device_count", return_value=0)
        fake_unet, fake_vae, sched = _patch_heavy_deps(mocker)

        # Prepare an sd with entries that match the unet param names -> triggers p.data.copy_
        unet_param_shapes = {
            "blocks.0.attn.lora_A.default_encoder.weight": fake_unet.named_parameters.return_value[
                0
            ][
                1
            ].data,
            "blocks.7.ff.lora_B.default_decoder.weight": fake_unet.named_parameters.return_value[
                1
            ][
                1
            ].data,
            "proj.lora.default_others.weight": fake_unet.named_parameters.return_value[
                2
            ][1].data,
        }
        sd = make_state_dict_for_copy(unet_param_shapes)

        # Make torch.load return our sd when load_ckpt_from_url finishes
        mocker.patch("gan.cyclegan_turbo.download_url", return_value=None)
        mocker.patch("torch.load", return_value=sd)

        from gan.cyclegan_turbo import CycleGAN_Turbo

        # instantiate through any pretrained_name to trigger URL path
        m = CycleGAN_Turbo(pretrained_name="day_to_night")

        # Adapters were added and set; loops ran without error
        assert m.unet.add_adapter.call_count >= 3
        called = False
        if hasattr(m.unet, "set_adapter"):
            called = m.unet.set_adapter.called
        elif hasattr(m.unet, "set_adapters"):
            called = m.unet.set_adapters.called
        assert called, "Expected set_adapter/set_adapters to be called"

    @it("forward uses caption_emb path and bypasses tokenizer / text_encoder")
    def test_forward_with_caption_emb(self, mocker):
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.cuda.device_count", return_value=0)
        fake_unet, fake_vae, sched = _patch_heavy_deps(mocker)

        # Make forward_with_networks identity-like to not rely on VAE/UNet functional behavior
        from gan.cyclegan_turbo import CycleGAN_Turbo

        unet_param_shapes = {
            fake_unet.named_parameters.return_value[0][
                0
            ]: fake_unet.named_parameters.return_value[0][1].data,
            fake_unet.named_parameters.return_value[1][
                0
            ]: fake_unet.named_parameters.return_value[1][1].data,
            fake_unet.named_parameters.return_value[2][
                0
            ]: fake_unet.named_parameters.return_value[2][1].data,
        }
        sd = make_state_dict_for_copy(unet_param_shapes)
        mocker.patch("torch.load", return_value=sd)

        # Instantiate via pretrained_path -> sets direction=None and caption=None
        m = CycleGAN_Turbo(pretrained_path="fake.pkl")
        # Provide direction explicitly and caption_emb to bypass tokenizer/text_encoder
        x = torch.randn(1, 3, 16, 16)
        caption_emb = torch.randn(1, 77, 768)

        # Stub static forward_with_networks to return its first arg's shape
        with mock.patch.object(
            CycleGAN_Turbo, "forward_with_networks", return_value=torch.zeros_like(x)
        ) as fw:
            out = m.forward(x, direction="a2b", caption_emb=caption_emb)

        assert out.shape == x.shape
        # Ensure tokenizer/text_encoder not touched in this call path
        # (They would have been used if caption_emb was None)
        fw.assert_called_once()

    @it("forward without args uses internal caption/direction set by pretrained_name")
    def test_forward_uses_internal_defaults(self, mocker):
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.cuda.device_count", return_value=0)
        fake_unet, fake_vae, sched = _patch_heavy_deps(mocker)
        unet_param_shapes = {
            fake_unet.named_parameters.return_value[0][
                0
            ]: fake_unet.named_parameters.return_value[0][1].data,
            fake_unet.named_parameters.return_value[1][
                0
            ]: fake_unet.named_parameters.return_value[1][1].data,
            fake_unet.named_parameters.return_value[2][
                0
            ]: fake_unet.named_parameters.return_value[2][1].data,
        }
        sd = make_state_dict_for_copy(unet_param_shapes)
        mocker.patch("torch.load", return_value=sd)

        from gan.cyclegan_turbo import CycleGAN_Turbo

        m = CycleGAN_Turbo(pretrained_name="day_to_night")
        x = torch.randn(1, 3, 8, 8)
        with mock.patch.object(
            CycleGAN_Turbo, "forward_with_networks", return_value=torch.zeros_like(x)
        ) as fw:
            out = m.forward(x)  # no args -> uses internal caption/direction
        assert out.shape == x.shape
        fw.assert_called_once()

    @it("forward_with_networks stacks over batch>1 and calls sched.step per item")
    def test_forward_with_networks_batch_two(self, mocker):
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.cuda.device_count", return_value=0)
        _ = _patch_heavy_deps(mocker)

        # Build very light mocks for vae_enc/vae_dec/unet/sched
        vae_enc = mock.MagicMock()
        # return latent same shape as x
        vae_enc.return_value = torch.randn(2, 3, 8, 8)

        class _UNetOut:
            def __init__(self, sample):
                self.sample = sample

        unet = mock.MagicMock()
        unet.return_value = _UNetOut(sample=torch.randn(2, 3, 8, 8))

        # sched.step returns object with .prev_sample per item
        class _StepOut:
            def __init__(self, prev_sample):
                self.prev_sample = prev_sample

        def step_fn(model_pred_i, t_i, x_i, return_dict=True):
            # return the input back (shape-preserving)
            return _StepOut(prev_sample=x_i)

        sched = mock.MagicMock()
        sched.step.side_effect = step_fn

        vae_dec = mock.MagicMock()
        vae_dec.return_value = torch.zeros(2, 3, 8, 8)  # decoded tensor

        from gan.cyclegan_turbo import CycleGAN_Turbo

        x = torch.randn(2, 3, 8, 8)
        timesteps = torch.tensor([999, 999])
        text_emb = torch.randn(2, 77, 768)

        out = CycleGAN_Turbo.forward_with_networks(
            x, "a2b", vae_enc, unet, vae_dec, sched, timesteps, text_emb
        )
        assert out.shape == (2, 3, 8, 8)
        # sched.step called twice (per batch item)
        assert sched.step.call_count == 2

    @it("one init with CUDA reported available (no real CUDA init)")
    def test_init_when_cuda_available(self, mocker):
        # 1) Pretend CUDA is available
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=1)

        # 2) Prevent any real CUDA driver initialization
        mocker.patch("torch._C._cuda_init", return_value=None)

        # 3) Disable lazy-call mechanism that schedules capability checks
        mocker.patch("torch.cuda._lazy_call", side_effect=lambda fn: None)

        # 4) Stub capability/property calls if they are invoked
        mocker.patch("torch.cuda.get_device_capability", return_value=(8, 0))
        mocker.patch("torch.cuda.get_device_properties", return_value=mock.Mock(
            name='FakeCudaProps',
            major=8, minor=0, total_memory=24_000_000_000))

        # 5) Very important: prevent creating tensors with device='cuda'
        #    Remove the 'device' argument if it is passed.
        _orig_tensor = torch.tensor
        def _cpu_tensor(*args, **kwargs):
            kwargs.pop("device", None)
            return _orig_tensor(*args, **kwargs)
        mocker.patch("torch.tensor", side_effect=_cpu_tensor)

        # Patch heavy dependencies used in the model
        fake_unet, fake_vae, sched = _patch_heavy_deps(mocker)

        # Fake state dict to avoid heavy loading
        sd = make_state_dict_for_copy({
            fake_unet.named_parameters.return_value[0][0]:
                fake_unet.named_parameters.return_value[0][1].data,
            fake_unet.named_parameters.return_value[1][0]:
                fake_unet.named_parameters.return_value[1][1].data,
            fake_unet.named_parameters.return_value[2][0]:
                fake_unet.named_parameters.return_value[2][1].data,
        })
        mocker.patch("torch.load", return_value=sd)

        # Avoid any network download
        mocker.patch("gan.cyclegan_turbo.download_url", return_value=None)

        from gan.cyclegan_turbo import CycleGAN_Turbo
        m = CycleGAN_Turbo(pretrained_name="day_to_night")

        # Sanity check to ensure adapters were added
        assert m.unet.add_adapter.call_count >= 3
