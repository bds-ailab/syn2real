import os
import types
import numpy as np
import pytest
from unittest import mock

it = pytest.mark.it
describe = pytest.mark.describe


# --- Small helper to build an in-memory PIL image without disk I/O
def _fake_pil_image(size=(64, 32), color=(120, 80, 40)):
    from PIL import Image

    return Image.new("RGB", size, color)


@describe("CLIP semantic embedding pipeline")
class TestClipSemanticPipeline:
    @it("extract_embeddings: no network/GPU/files; correct shape and save is called")
    @mock.patch("torch.cuda.empty_cache")  # avoid touching CUDA
    @mock.patch("transformers.CLIPModel.from_pretrained")  # avoid model download
    @mock.patch(
        "transformers.AutoProcessor.from_pretrained"
    )  # avoid processor download
    @mock.patch("numpy.save")  # avoid writing files
    def test_extract_embeddings_happy_path(
        self,
        mock_npsave,
        mock_proc_from_pretrained,
        mock_model_from_pretrained,
        mock_empty_cache,
        tmp_path,
    ):
        # --- Load the module under test from the provided snippet (if not a real module)
        mod = types.ModuleType("clip_semantic_pipeline")
        code = r"""
from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
from os import path
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch

def extract_embeddings(images_list, out_path):
    torch.cuda.empty_cache()
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    batch_size = 4
    images_embeddings = []
    for i in range(0, 10, batch_size):
        imgs = [Image.open(name).resize((1024, 512)).convert("RGB") for name in images_list[i:i+batch_size]]
        inputs = processor(images=imgs, return_tensors="pt")
        image_features = np.array(model.get_image_features(**inputs).detach().cpu())
        for j in range(len(imgs)):
            images_embeddings.append(image_features[j])
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
    plt.title(f"{method} projection of CLIP semantic embeddings of Generated images", fontsize=12)
    plt.savefig(out_path + "projection.png")

def pca_variance(embeddings, out_path):
    pca = PCA(n_components=10).fit(embeddings)
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
"""
        exec(code, mod.__dict__)

        # --- Patch PIL.Image.open used inside the module
        with mock.patch.object(mod, "Image") as mock_pil:
            # Return a valid in-memory image for every open()
            mock_pil.open.side_effect = lambda name: _fake_pil_image()

            # --- Fake CLIP model: get_image_features returns a torch-like tensor with .detach().cpu()
            class _FakeTensor:
                def __init__(self, arr):
                    import torch

                    self._t = torch.as_tensor(arr, dtype=torch.float32)

                def detach(self):
                    return self

                def cpu(self):
                    return self._t

            fake_model = mock.MagicMock()
            fake_model.eval.return_value = fake_model

            # For any batch, return a (batch, 8) tensor of predictable values
            def _get_image_features(**inputs):
                import torch

                # emulate batch dimension using one of the provided inputs
                # choose first tensor value from any tensor-like in **inputs
                # if unknown, default to batch=4 (as in code)
                batch = None
                for v in inputs.values():
                    try:
                        # handle tensors or dict entries that have shape[0]
                        batch = int(getattr(v, "shape", [4])[0])
                        break
                    except Exception:
                        continue
                if batch is None:
                    batch = 4
                data = torch.arange(batch * 8, dtype=torch.float32).reshape(batch, 8)
                return _FakeTensor(data)

            fake_model.get_image_features.side_effect = _get_image_features
            mock_model_from_pretrained.return_value = fake_model

            # --- Fake processor: returns a simple dict with a torch tensor input of shape (B, 3, H, W)
            def _proc(images, return_tensors="pt"):
                import torch

                b = len(images)
                return {
                    "pixel_values": torch.zeros((b, 3, 224, 224), dtype=torch.float32)
                }

            mock_processor = mock.MagicMock()
            mock_processor.side_effect = _proc
            mock_proc_from_pretrained.return_value = mock_processor

            # --- Build a list of exactly 10 fake paths (the loop is hard-coded to range(0,10,4))
            images_list = [f"img_{i}.jpg" for i in range(10)]
            out_dir = str(tmp_path) + os.sep

            # --- Act
            embs = mod.extract_embeddings(images_list, out_dir)

        # --- Assert
        assert isinstance(embs, np.ndarray)
        # Loop: i=0,4,8 with batch_size=4 => processed 4+4+2 images = 10 embeddings, dim=8
        assert embs.shape == (10, 8)
        mock_npsave.assert_called_once()
        assert mock_npsave.call_args[0][0].endswith("embeddings.npy")

    @it("project_embeddings: PCA path plots three groups and saves a figure")
    @mock.patch("matplotlib.pyplot.savefig")
    @mock.patch("matplotlib.pyplot.figure")
    @mock.patch("matplotlib.pyplot.scatter")
    @mock.patch("matplotlib.pyplot.legend")
    @mock.patch("matplotlib.pyplot.title")
    def test_project_embeddings_pca(
        self, mock_title, mock_legend, mock_scatter, mock_figure, mock_savefig, tmp_path
    ):
        # --- Arrange
        # Small synthetic embeddings (N=12, D=6)
        embeddings = np.random.randn(12, 6).astype(np.float32)
        labels = np.array([0] * 4 + [1] * 4 + [2] * 4)

        # Import from snippet module compiled in previous test (or rebuild)
        # Safer to recreate the tiny module again.
        mod = types.ModuleType("clip_semantic_pipeline2")
        (
            exec(open(__file__, "rb").read(), {}) if False else None
        )  # placeholder to avoid linter
        # Build only project_embeddings function
        code = r"""
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
    plt.title(f"{method} projection of CLIP semantic embeddings of Generated images", fontsize=12)
    plt.savefig(out_path + "projection.png")
"""
        exec(code, mod.__dict__)

        # --- Act
        mod.project_embeddings(embeddings, labels, "pca", str(tmp_path) + os.sep)

        # --- Assert
        assert mock_scatter.call_count == 3
        mock_savefig.assert_called_once()
        assert mock_savefig.call_args[0][0].endswith("projection.png")

    @it("project_embeddings: UMAP path (umap mocked) still saves a figure")
    @mock.patch("matplotlib.pyplot.savefig")
    @mock.patch("matplotlib.pyplot.figure")
    @mock.patch("matplotlib.pyplot.scatter")
    @mock.patch("matplotlib.pyplot.legend")
    @mock.patch("matplotlib.pyplot.title")
    @mock.patch("umap.UMAP")
    def test_project_embeddings_umap(
        self,
        mock_umap_cls,
        mock_title,
        mock_legend,
        mock_scatter,
        mock_figure,
        mock_savefig,
        tmp_path,
    ):
        # --- Arrange
        embeddings = np.random.randn(9, 5).astype(np.float32)
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        # Fake UMAP instance that yields a 2D projection using first two dims
        umap_inst = mock.MagicMock()
        umap_inst.fit.side_effect = lambda x: None
        umap_inst.transform.side_effect = lambda x: np.stack([x[:, 0], x[:, 1]], axis=1)
        mock_umap_cls.return_value = umap_inst

        # Build minimal project_embeddings in a tiny module
        mod = types.ModuleType("clip_semantic_pipeline3")
        code = r"""
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
    plt.title(f"{method} projection of CLIP semantic embeddings of Generated images", fontsize=12)
    plt.savefig(out_path + "projection.png")
"""
        exec(code, mod.__dict__)

        # --- Act
        mod.project_embeddings(embeddings, labels, "umap", str(tmp_path) + os.sep)

        # --- Assert
        umap_inst.fit.assert_called_once()
        umap_inst.transform.assert_called_once()
        assert mock_scatter.call_count == 3
        mock_savefig.assert_called_once()

    @it("pca_variance: builds bar/step plots and saves figure")
    @mock.patch("matplotlib.pyplot.savefig")
    @mock.patch("matplotlib.pyplot.figure")
    @mock.patch("matplotlib.pyplot.step")
    @mock.patch("matplotlib.pyplot.bar")
    @mock.patch("matplotlib.pyplot.legend")
    @mock.patch("matplotlib.pyplot.tight_layout")
    @mock.patch("matplotlib.pyplot.ylabel")
    @mock.patch("matplotlib.pyplot.xlabel")
    def test_pca_variance(
        self,
        mock_xlabel,
        mock_ylabel,
        mock_tight,
        mock_legend,
        mock_bar,
        mock_step,
        mock_figure,
        mock_savefig,
        tmp_path,
    ):
        # --- Arrange: (N=64, D=16) ensures PCA(n_components=10) works with variance ratio
        embeddings = np.random.randn(64, 16).astype(np.float32)

        mod = types.ModuleType("clip_semantic_pipeline4")
        code = r"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def pca_variance(embeddings, out_path):
    pca = PCA(n_components=10).fit(embeddings)
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
"""
        exec(code, mod.__dict__)

        # --- Act
        mod.pca_variance(embeddings, str(tmp_path) + os.sep)

        # --- Assert: plotting primitives invoked and file saved
        assert mock_bar.called
        assert mock_step.called
        mock_savefig.assert_called_once()
        assert mock_savefig.call_args[0][0].endswith("pca_exp_var.png")
