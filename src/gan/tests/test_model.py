import pytest
from unittest import mock
import torch
import torch.nn as nn
import os
import tempfile
import shutil
from PIL import Image

it = pytest.mark.it
describe = pytest.mark.describe
skip = pytest.mark.skip


class FnModule(nn.Module):
    """Wrapper for simple forward behaviour."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class Conv2dWithLE(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)

    def forward(self, x, le=None):
        return self.conv(x)


@describe("Test the script model")
class TestModel:

    @it("should create scheduler with 1 step on cuda")
    @mock.patch("gan.model.DDPMScheduler")
    def test_make_1step_sched(self, mock_sched_cls):
        from gan.model import make_1step_sched

        fake_sched = mock.Mock()
        fake_ac = mock.Mock()
        fake_ac.cuda.return_value = "ac_on_cuda"
        fake_sched.alphas_cumprod = fake_ac
        mock_sched_cls.from_pretrained.return_value = fake_sched

        out = make_1step_sched()

        mock_sched_cls.from_pretrained.assert_called_once_with(
            "stabilityai/sd-turbo", subfolder="scheduler"
        )
        fake_sched.set_timesteps.assert_called_once_with(1, device="cuda")
        assert out.alphas_cumprod == "ac_on_cuda"
        assert out is fake_sched

    @it("encoder: should collect skip activations and return final tensor")
    def test_my_vae_encoder_fwd(self):
        from gan.model import my_vae_encoder_fwd

        class EncSelf(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_in = nn.Conv2d(3, 8, 3, padding=1)
                self.down_blocks = nn.ModuleList(
                    [nn.Conv2d(8, 8, 3, padding=1) for _ in range(3)]
                )
                self.mid_block = nn.Conv2d(8, 8, 3, padding=1)
                self.conv_norm_out = nn.Identity()
                self.conv_act = nn.ReLU()
                self.conv_out = nn.Conv2d(8, 4, 1)
                self.current_down_blocks = []

        enc = EncSelf()
        x = torch.randn(2, 3, 16, 16)
        y = my_vae_encoder_fwd(enc, x)
        assert y.shape == (2, 4, 16, 16)
        assert len(enc.current_down_blocks) == 3

    @it("decoder: should use skip connections when ignore_skip=False")
    def test_my_vae_decoder_fwd_with_skips(self):
        from gan.model import my_vae_decoder_fwd

        C = 8
        incoming_skips = [torch.randn(2, C, 16, 16) for _ in range(4)]

        class DecSelf(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_in = nn.Conv2d(C, C, 3, padding=1)

                self.up_blocks = nn.ModuleList(
                    [
                        FnModule(lambda x, le=None: x),
                        Conv2dWithLE(C, C),
                        FnModule(lambda x, le=None: x),
                        FnModule(lambda x, le=None: x),
                    ]
                )
                self.mid_block = FnModule(lambda x, le=None: x)
                self.conv_norm_out = FnModule(lambda x, le=None: x)
                self.conv_act = nn.ReLU()
                self.conv_out = nn.Conv2d(C, 3, 1)
                self.skip_conv_1 = nn.Conv2d(C, C, 1, bias=False)
                self.skip_conv_2 = nn.Conv2d(C, C, 1, bias=False)
                self.skip_conv_3 = nn.Conv2d(C, C, 1, bias=False)
                self.skip_conv_4 = nn.Conv2d(C, C, 1, bias=False)
                self.incoming_skip_acts = incoming_skips
                self.ignore_skip = False
                self.gamma = 1.0

        dec = DecSelf()
        x = torch.randn(2, C, 16, 16)
        out = my_vae_decoder_fwd(dec, x)
        assert out.shape == (2, 3, 16, 16)

    @it("decoder: should skip skip-connections when ignore_skip=True")
    def test_my_vae_decoder_fwd_without_skips(self):
        from gan.model import my_vae_decoder_fwd

        C = 8

        class DecSelf(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_in = nn.Conv2d(C, C, 3, padding=1)
                self.up_blocks = nn.ModuleList(
                    [
                        Conv2dWithLE(C, C),
                        FnModule(lambda x, le=None: x),
                        FnModule(lambda x, le=None: x),
                        FnModule(lambda x, le=None: x),
                    ]
                )
                self.mid_block = FnModule(lambda x, le=None: x)
                self.conv_norm_out = FnModule(lambda x, le=None: x)
                self.conv_act = nn.ReLU()
                self.conv_out = nn.Conv2d(C, 3, 1)
                self.skip_conv_1 = nn.Conv2d(C, C, 1, bias=False)
                self.skip_conv_2 = nn.Conv2d(C, C, 1, bias=False)
                self.skip_conv_3 = nn.Conv2d(C, C, 1, bias=False)
                self.skip_conv_4 = nn.Conv2d(C, C, 1, bias=False)
                self.incoming_skip_acts = [torch.randn(2, C, 16, 16) for _ in range(4)]
                self.ignore_skip = True
                self.gamma = 1.0

        dec = DecSelf()
        x = torch.randn(2, C, 16, 16)
        out = my_vae_decoder_fwd(dec, x)
        assert out.shape == (2, 3, 16, 16)

    @it("should download file when missing")
    @mock.patch("gan.model.tqdm")
    @mock.patch("gan.model.requests.get")
    @mock.patch("os.path.exists", return_value=False)
    def test_download_url_when_missing(
        self, mock_exists, mock_get, mock_tqdm, tmp_path
    ):
        from gan.model import download_url

        chunks = [b"hi", b"there"]
        total_bytes = sum(len(c) for c in chunks)

        class FakeResp:
            headers = {"content-length": str(total_bytes)}

            def iter_content(self, block_size):
                yield from chunks

        mock_get.return_value = FakeResp()
        bar = mock.Mock()
        bar.n = total_bytes
        mock_tqdm.return_value = bar

        outf = tmp_path / "file.bin"
        download_url("http://x", str(outf))

        assert outf.read_bytes() == b"hithere"
        bar.update.assert_called()
        bar.close.assert_called_once()

    @it("should skip download if file exists")
    @mock.patch("os.path.exists", return_value=True)
    @mock.patch("gan.model.requests.get")
    def test_download_url_skip(self, mock_get, mock_exists, tmp_path):
        from gan.model import download_url

        outf = tmp_path / "f.bin"
        outf.write_bytes(b"x")
        download_url("http://x", str(outf))
        mock_get.assert_not_called()
