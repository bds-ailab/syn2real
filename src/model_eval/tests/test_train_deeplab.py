import os
import types
import numpy as np
from PIL import Image
import torch
import pytest
from unittest import mock

import model_eval.train_deeplab as mod


@pytest.mark.describe(
    "TestTrainDeepLab (tests-only fixes, no changes to production code)"
)
class TestTrainDeepLab:
    # ------------------------
    # Helpers
    # ------------------------
    @staticmethod
    def _write_cityscapes_like(root, n=4, size=(32, 64)):
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        h, w = size
        for i in range(n):
            img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
            Image.fromarray(img, mode="RGB").save(root / "images" / f"img_{i:03d}.png")

            # label in [0..33], uint8, mode 'L'
            lbl = np.random.randint(0, 34, size=(h, w), dtype=np.uint8)
            Image.fromarray(lbl, mode="L").save(root / "labels" / f"lbl_{i:03d}.png")

    @staticmethod
    def _make_scaled_mask(batch, height, width, num_classes):
        """
        Create a [B,1,H,W] float mask where values are k/255 so that
        (mask * 255).long() == k in [0 .. num_classes-1].
        """
        k = torch.randint(low=0, high=num_classes, size=(batch, 1, height, width))
        return k.float() / 255.0

    # ------------------------
    # Individual tests
    # ------------------------

    @pytest.mark.it("get_id raises ValueError when no digits are present")
    def test_get_id(self):
        with pytest.raises(ValueError):
            mod.get_id("no_digits_here.png")
        assert mod.get_id("img_0123.png") == 123

    @pytest.mark.it("CityscapesDataset.__getitem__ and reduce_labels mapping behavior")
    def test_dataset_getitem_and_reduce(self, tmp_path):
        root = tmp_path / "ds"
        self._write_cityscapes_like(root, n=1, size=(8, 8))

        # IMPORTANT: use a *valid Cityscapes ID not in reduced evaluation set* (e.g., 3)
        # reduce_labels loops over 0..33 and only remaps those; 99 would be untouched.
        lbl = np.full((8, 8), 3, dtype=np.uint8)  # 'out of roi' (not evaluated)
        Image.fromarray(lbl, mode="L").save(root / "labels" / "lbl_000.png")

        ds = mod.CityscapesDataset(
            root=str(root),
            labels=mod.reduced_labels,
            transforms=(torch.nn.Identity(), torch.nn.Identity()),
            shuffle=False,
            select_range=None,
            seed=0,
        )
        sample = ds[0]
        reduced = ds.reduce_labels(sample["mask"])
        arr = np.array(reduced, dtype=np.uint8)
        # All 3's must be mapped to 0 (ignored / background)
        assert (arr == 0).all()

    @pytest.mark.it("eval_model handles one batch: fake model logits resize to masks")
    def test_eval_model_one_batch(self, mocker):
        class FakeModel(torch.nn.Module):
            def eval(self):
                return self

            def forward(self, x):
                B, _, H, W = x.shape
                return {"out": torch.randn(B, len(mod.labels), H, W)}

        B, H, W = 2, 16, 32
        # IMPORTANT: provide masks scaled in [0,1] so (mask*255).long() is a valid class id
        mask_scaled = self._make_scaled_mask(B, H, W, num_classes=len(mod.labels))
        batch = {"image": torch.rand(B, 3, H, W), "mask": mask_scaled}
        dls = {"Test": [batch]}

        fake_metric = mock.Mock()
        fake_metric.compute.return_value = {
            "mean_iou": 0.5,
            "mean_accuracy": 0.6,
            "overall_accuracy": 0.7,
            "frequency_weighted_iou": 0.55,
            "per_category_iou": torch.zeros(len(mod.labels)),
            "per_category_accuracy": torch.zeros(len(mod.labels)),
        }

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        id2label = {label.id: label.name for label in mod.labels}

        loss, mean_iou, results = mod.eval_model(
            model=FakeModel(),
            dataloaders=dls,
            metric=fake_metric,
            criterion=loss_fn,
            id2label=id2label,
        )
        assert isinstance(loss, float)
        assert "iou_mean" in results

    @pytest.mark.it(
        "get_dataloader does not crash with small dataset (patch select_range)"
    )
    def test_get_dataloader(self, tmp_path, mocker):
        train_root = tmp_path / "train"
        test_root = tmp_path / "test"
        self._write_cityscapes_like(train_root, n=2)
        self._write_cityscapes_like(test_root, n=2)

        original_cls = mod.CityscapesDataset

        def _patched_cityscapes(
            root, labels, transforms, shuffle=True, select_range=None, seed=0
        ):
            return original_cls(
                root, labels, transforms, shuffle=shuffle, select_range=None, seed=seed
            )

        mocker.patch.object(mod, "CityscapesDataset", side_effect=_patched_cityscapes)

        dls = mod.get_dataloader(
            train_dataset=str(train_root),
            test_dataset=str(test_root),
            batch_size=2,
            n_workers=0,
        )
        batch = next(iter(dls["Train"]))
        assert "image" in batch and "mask" in batch

    @pytest.mark.it("main executes end-to-end with all heavy parts mocked")
    def test_main_happy_path(self, tmp_path, mocker):
        train_root = tmp_path / "train"
        test_root = tmp_path / "test"
        exp_root = tmp_path / "exp"
        self._write_cityscapes_like(train_root, n=2)
        self._write_cityscapes_like(test_root, n=2)

        original_cls = mod.CityscapesDataset

        def _patched_cityscapes(
            root, labels, transforms, shuffle=True, select_range=None, seed=0
        ):
            return original_cls(
                root, labels, transforms, shuffle=shuffle, select_range=None, seed=seed
            )

        mocker.patch.object(mod, "CityscapesDataset", side_effect=_patched_cityscapes)

        class FakeModel(torch.nn.Module):
            def train(self, *a, **k):
                return self

            def to(self, *a, **k):
                return self

            def forward(self, x):
                B, _, H, W = x.shape
                return {"out": torch.randn(B, len(mod.labels), H, W)}

        mocker.patch(
            "model_eval.train_deeplab.createDeepLabv3", return_value=FakeModel()
        )

        fake_metric = mock.Mock()
        fake_metric.compute.return_value = {
            "mean_iou": 0.2,
            "mean_accuracy": 0.3,
            "overall_accuracy": 0.4,
            "frequency_weighted_iou": 0.25,
            "per_category_iou": torch.zeros(len(mod.labels)),
            "per_category_accuracy": torch.zeros(len(mod.labels)),
        }
        mocker.patch("model_eval.train_deeplab.evaluate.load", return_value=fake_metric)

        # IMPORTANT: ensure masks produce valid class IDs after *255 and .long()
        def _dl_factory(ds, **kw):
            B, H, W = 2, 16, 32
            sample = {
                "image": torch.rand(B, 3, H, W),
                "mask": self._make_scaled_mask(B, H, W, num_classes=len(mod.labels)),
            }
            return [sample]

        mocker.patch("torch.utils.data.DataLoader", side_effect=_dl_factory)

        # 5) Optimizer/scaler/amp as no-ops
        mocker.patch(
            "torch.optim.Adam",
            return_value=mock.Mock(zero_grad=mock.Mock(), step=mock.Mock()),
        )

        # Return a GradScaler stub whose `scale(loss)` returns an object with `.backward()` no-op
        class _Scaled:
            def __init__(self, loss):
                self.loss = loss

            def backward(self):
                return None

        mocker.patch(
            "model_eval.train_deeplab.GradScaler",
            return_value=mock.Mock(
                scale=lambda loss: _Scaled(loss),
                step=lambda *a, **k: None,
                update=lambda *a, **k: None,
            ),
        )

        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, *a):
                return False

        mocker.patch("model_eval.train_deeplab.autocast", return_value=_Ctx())
        mocker.patch("model_eval.train_deeplab.autocast", return_value=_Ctx())

        mocker.patch("model_eval.train_deeplab.tqdm", side_effect=lambda x: x)

        args = types.SimpleNamespace(
            in_train_dataset=str(train_root),
            in_test_dataset=str(test_root),
            exp_folder=str(exp_root),
            batch_size=2,
            num_workers=0,
            epochs=1,
            learning_rate=1e-4,
        )

        mod.main(args)
        assert os.path.isdir(args.exp_folder)
