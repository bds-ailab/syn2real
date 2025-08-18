import pytest
import numpy as np
import torch

import model_eval.mmd as mod


@pytest.mark.describe("Gaussian RBF MMD (mmd.py)")
class TestMMD:

    @pytest.mark.it("returns a PyTorch scalar (0-dim tensor) and is non-negative")
    def test_output_type_and_nonnegativity(self):
        # Small random inputs, identical seeds for reproducibility
        rng = np.random.default_rng(0)
        x = rng.standard_normal((8, 4)).astype(np.float32)
        y = rng.standard_normal((8, 4)).astype(np.float32)

        out = mod.mmd(x, y)
        # Check torch scalar
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0, "Expected a 0-dim torch scalar tensor"
        # MMD should be >= 0
        assert float(out) >= 0.0

    @pytest.mark.it("is zero (up to numerical precision) when x == y")
    def test_zero_when_same(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((10, 3)).astype(np.float64)
        out = mod.mmd(x, x)
        assert torch.isfinite(out)
        # Allow tiny numerical noise
        assert float(out) == pytest.approx(0.0, abs=1e-8)

    @pytest.mark.it("is symmetric: mmd(x, y) == mmd(y, x)")
    def test_symmetry(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((12, 6)).astype(np.float32)
        y = rng.standard_normal((12, 6)).astype(np.float32)
        dxy = float(mod.mmd(x, y))
        dyx = float(mod.mmd(y, x))
        assert dxy == pytest.approx(dyx, rel=1e-7, abs=1e-10)

    @pytest.mark.it("increases as distributions move farther apart")
    def test_monotonic_trend_with_separation(self):
        rng = np.random.default_rng(3)
        base = rng.standard_normal((16, 5)).astype(np.float32)
        # Create shifted versions of base
        y_small = base + 0.1
        y_big = base + 2.0

        d_small = float(mod.mmd(base, y_small))
        d_big = float(mod.mmd(base, y_big))
        assert d_big > d_small

    @pytest.mark.it("works with both float32 and float64 numpy inputs")
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_handling(self, dtype):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((7, 3)).astype(dtype)
        y = rng.standard_normal((7, 3)).astype(dtype)
        out = mod.mmd(x, y)
        assert torch.is_tensor(out)
        assert out.ndim == 0
        assert torch.isfinite(out)

    @pytest.mark.it("raises or fails clearly when shapes are incompatible")
    @pytest.mark.it(
        "accepts different numbers of samples (n_x != n_y) and stays finite"
    )
    def test_different_sample_count_ok(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((8, 4)).astype(np.float32)
        y = rng.standard_normal((9, 4)).astype(np.float32)  # different n but same dim
        out = mod.mmd(x, y)
        assert torch.is_tensor(out) and out.ndim == 0
        assert torch.isfinite(out)

    @pytest.mark.it(
        "matches a manual small-case calculation when sigma/scale are patched"
    )
    def test_matches_manual_when_constants_patched(self, monkeypatch):
        # Use a tiny example where we can compute expected value numerically in numpy
        # Patch constants to stable values to avoid scaling confusion
        monkeypatch.setattr(mod, "_SIGMA", 1.0, raising=True)
        monkeypatch.setattr(mod, "_SCALE", 1.0, raising=True)

        # Two 2D points each
        x = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        y = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float64)

        # Gaussian RBF with gamma=1/(2*sigma^2)=1/2
        gamma = 1.0 / (2.0 * (mod._SIGMA**2))  # = 0.5

        # Helper to compute kernel matrix element-wise using numpy
        def rbf(a, b):
            # pairwise squared distances
            d2 = (
                np.sum(a**2, axis=1)[:, None]
                + np.sum(b**2, axis=1)[None, :]
                - 2 * a @ b.T
            )
            return np.exp(-gamma * d2)

        k_xx = rbf(x, x).mean()
        k_yy = rbf(y, y).mean()
        k_xy = rbf(x, y).mean()

        expected = k_xx + k_yy - 2.0 * k_xy

        out = float(mod.mmd(x, y))
        assert out == pytest.approx(expected, rel=1e-10, abs=1e-12)

    @pytest.mark.it("is stable for zero-variance inputs (identical vectors)")
    def test_zero_variance_inputs(self):
        x = np.zeros((5, 3), dtype=np.float32)
        y = np.zeros((5, 3), dtype=np.float32)
        out = mod.mmd(x, y)
        assert float(out) == pytest.approx(0.0, abs=1e-8)

    @pytest.mark.it("handles larger embedding dimensions without crashing")
    def test_large_embedding_dim(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal((32, 128)).astype(np.float32)
        y = rng.standard_normal((32, 128)).astype(np.float32)
        out = mod.mmd(x, y)
        assert torch.isfinite(out)
