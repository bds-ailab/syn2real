import os
import io
import json
import random
import types
import pytest
import torch
from PIL import Image
from unittest import mock
import argparse

it = pytest.mark.it
describe = pytest.mark.describe

# ---------------------------
# Helpers
# ---------------------------


def _make_dummy_image(size=(64, 64), color=(128, 200, 10)):
    img = Image.new("RGB", size, color)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return Image.open(bio).convert("RGB")


def _write_png(path, size=(64, 64)):
    img = _make_dummy_image(size=size)
    img.save(path)


class _DummyTokenizer:
    """Mimics HF tokenizer minimal API used by the datasets."""

    def __init__(self, max_len=77):
        self.model_max_length = max_len

    class _Tokens:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, max_length, padding, truncation, return_tensors):
        # return (1, max_length) tensor
        return _DummyTokenizer._Tokens(torch.randint(0, 100, (1, max_length)))


# ---------------------------
# Imports under test
# ---------------------------

from gan.training_utils import (
    parse_args_paired_training,
    parse_args_unpaired_training,
    build_transform,
    PairedDataset,
    UnpairedDataset,
)

# ===========================
# Tests parse_args_*
# ===========================


@describe("CLI parsers")
class TestParsers:

    @it("parse_args_paired_training: minimal happy path")
    def test_parse_args_paired_minimal(self, tmp_path):
        args = [
            "--dataset_folder",
            str(tmp_path / "data"),
            "--output_dir",
            str(tmp_path / "out"),
        ]
        ns = parse_args_paired_training(args)
        assert ns.dataset_folder == str(tmp_path / "data")
        assert ns.output_dir == str(tmp_path / "out")
        # defaults we rely on elsewhere
        assert ns.train_image_prep == "resized_crop_512"
        assert ns.adam_beta1 == 0.9

    @it("parse_args_unpaired_training: minimal happy path")
    def test_parse_args_unpaired_minimal(self, tmp_path):
        fake_ns = argparse.Namespace(
            seed=42,
            gan_disc_type="vagan_clip",
            gan_loss_type="multilevel_sigmoid",
            lambda_gan=0.5,
            lambda_idt=1.0,
            lambda_cycle=1.0,
            lambda_cycle_lpips=10.0,
            lambda_idt_lpips=1.0,
            dataset_folder=str(tmp_path / "data"),
            train_img_prep="resize_256",
            val_img_prep="resize_256x256",
            dataloader_num_workers=0,
            train_batch_size=4,
            max_train_epochs=100,
            max_train_steps=None,
            pretrained_model_name_or_path="stabilityai/sd-turbo",
            revision=None,
            variant=None,
            lora_rank_unet=128,
            lora_rank_vae=4,
            viz_freq=20,
            output_dir=str(tmp_path / "out"),
            report_to="wandb",
            tracker_project_name="proj",
            validation_steps=500,
            validation_num_images=-1,
            checkpointing_steps=500,
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_weight_decay=1e-2,
            adam_epsilon=1e-8,
            max_grad_norm=10.0,
            lr_scheduler="constant",
            lr_warmup_steps=500,
            lr_num_cycles=1,
            lr_power=1.0,
            gradient_accumulation_steps=1,
            allow_tf32=False,
            gradient_checkpointing=False,
            enable_xformers_memory_efficient_attention=False,
        )

        with mock.patch("argparse.ArgumentParser.parse_args", return_value=fake_ns):
            ns = parse_args_unpaired_training()

        assert ns.dataset_folder.endswith("data")
        assert ns.output_dir.endswith("out")
        assert ns.train_img_prep == "resize_256"
        assert ns.val_img_prep == "resize_256x256"


# ===========================
# Tests build_transform
# ===========================


@describe("build_transform")
class TestBuildTransform:

    @it("returns a Compose for resized_crop_512 and transforms an image")
    def test_resized_crop_512(self):
        T = build_transform("resized_crop_512")
        img = _make_dummy_image((600, 700))
        out = T(img)
        assert out.size == (512, 512)

    @it("resize_286_randomcrop_256x256_hflip produces 256x256")
    def test_randomcrop_pipeline(self):
        T = build_transform("resize_286_randomcrop_256x256_hflip")
        img = _make_dummy_image((300, 300))
        out = T(img)
        assert out.size == (256, 256)

    @it("resize_256 and resize_256x256 produce 256x256")
    def test_resize_256_variants(self):
        for key in ("resize_256", "resize_256x256"):
            T = build_transform(key)
            img = _make_dummy_image((300, 400))
            out = T(img)
            assert out.size == (256, 256)

    @it("resize_512 and resize_512x512 produce 512x512")
    def test_resize_512_variants(self):
        for key in ("resize_512", "resize_512x512"):
            T = build_transform(key)
            img = _make_dummy_image((700, 400))
            out = T(img)
            assert out.size == (512, 512)

    @it("no_resize returns identity")
    def test_no_resize(self):
        T = build_transform("no_resize")
        img = _make_dummy_image((123, 234))
        out = T(img)
        assert out.size == img.size

    @it("pers_resize resizes to 512x1024")
    def test_pers_resize(self):
        T = build_transform("pers_resize")
        img = _make_dummy_image((300, 300))
        out = T(img)
        assert out.size == (1024, 512) or out.size == (512, 1024)  # PIL uses (W,H)


# ===========================
# Tests PairedDataset
# ===========================


@describe("PairedDataset")
class TestPairedDataset:

    @it(
        "Must initialize the paired dataset object for loading and transforming paired data samples from test split."
    )
    def test_paired_train_split(self, tmp_path, monkeypatch):
        # --- structure ---
        root = tmp_path / "paired"
        (root / "train_A").mkdir(parents=True, exist_ok=True)
        (root / "train_B").mkdir(parents=True, exist_ok=True)

        # two images with same names for A and B
        for name in ["a.png", "b.png"]:
            _write_png(root / "train_A" / name, size=(64, 80))
            _write_png(root / "train_B" / name, size=(64, 80))

        captions = {"a.png": "caption A", "b.png": "caption B"}
        with open(root / "train_prompts.json", "w") as f:
            json.dump(captions, f)

        tok = _DummyTokenizer(max_len=16)
        ds = PairedDataset(
            dataset_folder=str(root),
            split="train",
            image_prep="resize_256",
            tokenizer=tok,
        )

        assert len(ds) == 2

        sample = ds[0]
        # expected keys
        assert set(sample.keys()) == {
            "output_pixel_values",
            "conditioning_pixel_values",
            "caption",
            "input_ids",
        }
        # shapes (C,H,W) after resize_256
        assert tuple(sample["conditioning_pixel_values"].shape) == (3, 256, 256)
        assert tuple(sample["output_pixel_values"].shape) == (3, 256, 256)
        # normalization of output: roughly [-1,1]
        out = sample["output_pixel_values"]
        assert out.min() >= -1.01 and out.max() <= 1.01
        # tokenizer output shape
        assert tuple(sample["input_ids"].shape) == (1, tok.model_max_length)

    @it(
        "Must initialize the paired dataset object for loading and transforming paired data samples from test split."
    )
    def test_paired_test_split(self, tmp_path):
        root = tmp_path / "paired2"
        (root / "test_A").mkdir(parents=True, exist_ok=True)
        (root / "test_B").mkdir(parents=True, exist_ok=True)

        for name in ["x.png"]:
            _write_png(root / "test_A" / name, size=(48, 48))
            _write_png(root / "test_B" / name, size=(48, 48))

        captions = {"x.png": "hello world"}
        with open(root / "test_prompts.json", "w") as f:
            json.dump(captions, f)

        tok = _DummyTokenizer(max_len=8)
        ds = PairedDataset(
            dataset_folder=str(root),
            split="test",
            image_prep="no_resize",
            tokenizer=tok,
        )
        assert len(ds) == 1
        item = ds[0]
        assert item["caption"] == "hello world"
        # with no_resize, size stays close and then to_tensor -> (3,H,W)
        assert item["conditioning_pixel_values"].shape[0] == 3
        assert item["output_pixel_values"].shape[0] == 3


# ===========================
# Tests UnpairedDataset
# ===========================


@describe("UnpairedDataset")
class TestUnpairedDataset:

    @it("Must load unpaired data samples from two distinct domains")
    def test_unpaired_len(self, tmp_path):
        root = tmp_path / "unpaired"
        (root / "train_A").mkdir(parents=True, exist_ok=True)
        (root / "train_B").mkdir(parents=True, exist_ok=True)

        # 3 src images, 2 tgt images
        for i in range(3):
            _write_png(root / "train_A" / f"s{i}.png")
        for i in range(2):
            _write_png(root / "train_B" / f"t{i}.png")

        # fixed prompts
        (root / "fixed_prompt_a.txt").write_text("source prompt")
        (root / "fixed_prompt_b.txt").write_text("target prompt")

        tok = _DummyTokenizer(max_len=12)
        ds = UnpairedDataset(
            dataset_folder=str(root),
            split="train",
            image_prep="resize_256",
            tokenizer=tok,
        )
        # len = len(src)+len(tgt)
        assert len(ds) == 5

    @it(
        "Must load unpaired data samples from two distinct domains, with a specific size"
    )
    def test_unpaired_getitem(self, tmp_path, monkeypatch):
        root = tmp_path / "unpaired2"
        (root / "train_A").mkdir(parents=True, exist_ok=True)
        (root / "train_B").mkdir(parents=True, exist_ok=True)

        for i in range(2):
            _write_png(root / "train_A" / f"s{i}.png", size=(60, 70))
        for i in range(4):
            _write_png(root / "train_B" / f"t{i}.png", size=(70, 60))

        (root / "fixed_prompt_a.txt").write_text("src fixed")
        (root / "fixed_prompt_b.txt").write_text("tgt fixed")

        tok = _DummyTokenizer(max_len=10)
        ds = UnpairedDataset(
            dataset_folder=str(root),
            split="train",
            image_prep="resize_256x256",
            tokenizer=tok,
        )

        # force determinism for random.choice during the test
        with mock.patch("random.choice", side_effect=lambda seq: seq[0]):
            item0 = ds[
                0
            ]  # index < len(src) -> specific src, random tgt (mocked -> first)
            item3 = ds[
                3
            ]  # index >= len(src) -> random src (mocked -> first), random tgt (mocked -> first)

        # Check keys
        for it in (item0, item3):
            assert set(it.keys()) == {
                "pixel_values_src",
                "pixel_values_tgt",
                "caption_src",
                "caption_tgt",
                "input_ids_src",
                "input_ids_tgt",
            }
            # Shapes after resize_256x256 -> (3,256,256)
            assert tuple(it["pixel_values_src"].shape) == (3, 256, 256)
            assert tuple(it["pixel_values_tgt"].shape) == (3, 256, 256)
            # Normalization ~ [-1,1]
            assert (
                it["pixel_values_src"].min() >= -1.01
                and it["pixel_values_src"].max() <= 1.01
            )
            assert (
                it["pixel_values_tgt"].min() >= -1.01
                and it["pixel_values_tgt"].max() <= 1.01
            )
            # Captions and tokenized ids
            assert it["caption_src"] == "src fixed"
            assert it["caption_tgt"] == "tgt fixed"
            assert tuple(it["input_ids_src"].shape) == (1, tok.model_max_length)
            assert tuple(it["input_ids_tgt"].shape) == (1, tok.model_max_length)
