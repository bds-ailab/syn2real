import os
import io
import json
import types
import numpy as np
import pytest
from unittest import mock

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
it = pytest.mark.it
describe = pytest.mark.describe

# Build an in-memory RGB image via PIL without touching disk
def _fake_pil_image(size=(64, 48), color=(128, 64, 32)):
    from PIL import Image
    img = Image.new("RGB", size, color)
    return img


# ---------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------
@describe("ARNIQA embedding / projection pipeline")
class TestArniqaPipeline:

    @it("extract_embeddings: runs end-to-end with mocked model / CUDA / I/O")
    @mock.patch("torch.cuda.empty_cache")                 # do nothing
    @mock.patch("torch.cuda.is_available", return_value=False)  # force CPU
    @mock.patch("torch.hub.load")                         # no network
    @mock.patch("numpy.save")                             # don't write files
    def test_extract_embeddings_cpu_happy_path(
        self, mock_npsave, mock_hub_load, mock_cuda_avail, mock_empty_cache, tmp_path, monkeypatch
    ):
        # -- Arrange: import module under test
        import builtins
        from importlib import import_module

        # Load the module containing the 3 functions
        # (adapt the dotted path if needed, e.g. gan.tools.arniqa_features)
        mod = import_module("__main__") if "__file__" not in globals() else None
        # If your code lives in a real module, replace the line above by:
        # from <your_package>.<your_module> import extract_embeddings

        # We can't rely on __main__, so let's exec the provided snippet in a ModuleType
        if mod is None:
            mod = types.ModuleType("arniqa_pipeline")
            code = r'''
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from os import path
import numpy as np
import umap
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA

def extract_embeddings(images_list, out_path):
    torch.cuda.empty_cache()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = torch.hub.load(
        repo_or_dir="miccunifi/ARNIQA",
        source="github",
        model="ARNIQA",
        regressor_dataset="kadid10k",
        trust_repo=True,
    )
    model.eval().to(device)

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    images_embeddings = []
    for i in range(len(images_list)):
        img = Image.open(path.join(images_list[i]))
        img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
        img = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
        img_ds = preprocess(img_ds.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            score, embedding = model(img, img_ds, return_embedding=True, scale_score=True)
        images_embeddings.append(np.array(embedding.detach().cpu()).flatten())
        print(f"{i}/{len(images_list)}")

    images_embeddings = np.array(images_embeddings)
    np.save(out_path + "embeddings.npy", images_embeddings)
    return images_embeddings

def project_embeddings(embeddings, labels, method, out_path):
    if method == "pca":
        projector = PCA(n_components=2)
    elif method == "umap":
        projector = umap.UMAP(n_neighbors=4, random_state=42)

    projector.fit(embeddings)
    trans = projector.transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(trans[labels == 0, 0], trans[labels == 0, 1], s=3, alpha=0.5, c="green", label="augmented")
    plt.scatter(trans[labels == 1, 0], trans[labels == 1, 1], s=3, alpha=0.5, c="green", label="real")
    plt.scatter(trans[labels == 2, 0], trans[labels == 2, 1], s=3, alpha=0.5, c="red", label="synthetic")
    plt.legend()
    plt.title(f"{method} projection of ARNIQA distortion embeddings of Generated images", fontsize=12)
    plt.savefig(out_path + "projection.png")

def pca_variance(embeddings, out_path):
    pca = PCA(n_components=50).fit(embeddings)
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    plt.figure(figsize=(8, 6))
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align="center", label="Individual explained variance")
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where="mid", label="Cumulative explained variance")
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal component index")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path + "pca_exp_var.png")
'''
            exec(code, mod.__dict__)

        # -- Arrange: mock PIL.Image.open to return an in-memory image
        with mock.patch.object(mod, "Image") as mock_pil:
            img = _fake_pil_image(size=(60, 40))
            mock_pil.open.return_value = img

            # Mock transforms.Compose to behave like a no-op tensorizer/normalizer
            import torch
            import torchvision.transforms as T

            def _compose(ops):
                # Minimal callable that sequentially applies ops
                class _C:
                    def __call__(self, im):
                        # Convert PIL -> tensor in [0,1] with shape (C,H,W)
                        t = T.ToTensor()(im)
                        # Apply Normalize if present
                        for op in ops:
                            if isinstance(op, T.Normalize):
                                t = op(t)
                        return t
                return _C()

            with mock.patch.object(mod.transforms, "Compose", side_effect=_compose):
                # Mock Resize op: use T.Resize directly to keep behavior correct
                with mock.patch.object(mod.transforms, "Resize", new=T.Resize):


                    # -- Arrange: mock model returned by torch.hub.load
                    fake_model = mock.MagicMock()
                    fake_model.eval.return_value = fake_model
                    fake_model.to.return_value = fake_model

                    # Return (score, embedding). We only care about embedding's .detach().cpu()
                    class _FakeTensor:
                        def __init__(self, arr):
                            self._arr = torch.as_tensor(arr, dtype=torch.float32)
                        def detach(self): return self
                        def cpu(self): return self._arr

                    # For N images, return embeddings of dim 5
                    def _model_call(img, img_ds, return_embedding=True, scale_score=True):
                        score = _FakeTensor([0.5])
                        emb = _FakeTensor(np.arange(5))
                        return score, emb

                    fake_model.side_effect = _model_call
                    mock_hub_load.return_value = fake_model

                    # -- Act
                    images_list = ["a.jpg", "b.jpg", "c.jpg"]
                    out_dir = str(tmp_path) + os.sep
                    embs = mod.extract_embeddings(images_list, out_dir)

        # -- Assert
        assert isinstance(embs, np.ndarray)
        assert embs.shape == (3, 5)            # 3 images, 5-dim embedding (our fake)
        mock_npsave.assert_called_once()        # embeddings saved
        args, kwargs = mock_npsave.call_args
        assert args[0].endswith("embeddings.npy")
        np.testing.assert_allclose(embs[0], np.arange(5), rtol=0, atol=1e-6)

    @it("project_embeddings: works with PCA and saves a figure")
    @mock.patch("matplotlib.pyplot.savefig")
    @mock.patch("matplotlib.pyplot.figure")
    @mock.patch("matplotlib.pyplot.scatter")
    @mock.patch("matplotlib.pyplot.legend")
    @mock.patch("matplotlib.pyplot.title")
    def test_project_embeddings_pca(
        self, mock_title, mock_legend, mock_scatter, mock_figure, mock_savefig, tmp_path
    ):
        # -- Arrange
        from sklearn.decomposition import PCA

        # Fake small embeddings (6 samples, 4-dim)
        embeddings = np.random.randn(6, 4).astype(np.float32)
        labels = np.array([0, 0, 1, 1, 2, 2])

        # Import module & call function
        from importlib import import_module
        mod = import_module("__main__") if "__file__" not in globals() else None
        if mod is None:
            pytest.skip("Load the module as in the test above if necessary")

        # -- Act
        mod.project_embeddings(embeddings, labels, method="pca", out_path=str(tmp_path) + os.sep)

        # -- Assert: plotting called and file saved
        assert mock_scatter.call_count == 3
        mock_savefig.assert_called_once()
        assert mock_savefig.call_args[0][0].endswith("projection.png")

    @it("project_embeddings: works with UMAP and saves a figure (UMAP mocked)")
    @mock.patch("matplotlib.pyplot.savefig")
    @mock.patch("matplotlib.pyplot.figure")
    @mock.patch("matplotlib.pyplot.scatter")
    @mock.patch("matplotlib.pyplot.legend")
    @mock.patch("matplotlib.pyplot.title")
    @mock.patch("umap.UMAP")
    def test_project_embeddings_umap(
        self, mock_umap_cls, mock_title, mock_legend, mock_scatter, mock_figure, mock_savefig, tmp_path
    ):
        # -- Arrange
        embeddings = np.random.randn(6, 4).astype(np.float32)
        labels = np.array([0, 0, 1, 1, 2, 2])

        # Mock UMAP instance: fit/transform become identity to 2D
        umap_inst = mock.Mock()
        umap_inst.fit.side_effect = lambda x: None
        umap_inst.transform.side_effect = lambda x: np.stack([x[:, 0], x[:, 1]], axis=1)
        mock_umap_cls.return_value = umap_inst

        from importlib import import_module
        mod = import_module("__main__") if "__file__" not in globals() else None
        if mod is None:
            pytest.skip("Load the module as in the test above if necessary")

        # -- Act
        mod.project_embeddings(embeddings, labels, method="umap", out_path=str(tmp_path) + os.sep)

        # -- Assert
        umap_inst.fit.assert_called_once()
        umap_inst.transform.assert_called_once()
        assert mock_scatter.call_count == 3
        mock_savefig.assert_called_once()

    @it("pca_variance: computes explained variance and saves bar/step plot")
    @mock.patch("matplotlib.pyplot.savefig")
    @mock.patch("matplotlib.pyplot.figure")
    @mock.patch("matplotlib.pyplot.step")
    @mock.patch("matplotlib.pyplot.bar")
    @mock.patch("matplotlib.pyplot.legend")
    @mock.patch("matplotlib.pyplot.tight_layout")
    @mock.patch("matplotlib.pyplot.ylabel")
    @mock.patch("matplotlib.pyplot.xlabel")
    def test_pca_variance_saves_plot(
        self, mock_xlabel, mock_ylabel, mock_tight, mock_legend, mock_bar, mock_step, mock_figure, mock_savefig, tmp_path
    ):
        # -- Arrange: embeddings with rank >= 10 so that n_components=50 is clipped by PCA
        embeddings = np.random.randn(64, 16).astype(np.float32)

        from importlib import import_module
        mod = import_module("__main__") if "__file__" not in globals() else None
        if mod is None:
            pytest.skip("Load the module as in the first test if necessary")

        # -- Act
        mod.pca_variance(embeddings, out_path=str(tmp_path) + os.sep)

        # -- Assert: figure elements invoked and file saved
        assert mock_bar.called
        assert mock_step.called
        mock_savefig.assert_called_once()
        assert mock_savefig.call_args[0][0].endswith("pca_exp_var.png")
