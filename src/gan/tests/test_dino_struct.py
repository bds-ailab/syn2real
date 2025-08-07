import pytest
from unittest import mock
import torch
from gan.dino_struct import DinoStructureLoss, VitExtractor, attn_cosine_sim

it = pytest.mark.it
describe = pytest.mark.describe
skip = pytest.mark.skip


@describe("TestDinoStruct")
class TestDinoStruct:

    @pytest.fixture
    def dummy_img(self):
        # Return a dummy image tensor of shape (B, C, H, W)
        return torch.randn(1, 3, 224, 224)

    @pytest.fixture
    def extractor_mock(self):
        extractor = mock.create_autospec(VitExtractor, instance=True)
        # Fake key self-similarity: (1, 1, N, N) tensor
        fake_ssim = torch.randn(1, 1, 197, 197)
        extractor.get_keys_self_sim_from_input.return_value = fake_ssim
        return extractor

    @it("it must return identity when input is identity matrix")
    def test_attn_cosine_sim_identity(self):
        x = torch.eye(4).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 4, 4)
        expected = torch.eye(4).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
        sim = attn_cosine_sim(x)
        assert torch.allclose(
            sim, expected, atol=1e-6
        ), "Cosine sim of identity should be identity matrix"

    @it("it must calculate the global SSIM loss with a mocked extractor")
    def test_calculate_global_ssim_loss_with_mocked_extractor(
        self, dummy_img, extractor_mock
    ):
        # Replace the real extractor with a mocked one
        structure_loss = DinoStructureLoss()
        structure_loss.extractor = extractor_mock

        output_img = dummy_img.clone()
        loss = structure_loss.calculate_global_ssim_loss([output_img], [dummy_img])

        # Should call extractor twice (once for input, once for output)
        assert extractor_mock.get_keys_self_sim_from_input.call_count == 2
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Should be scalar

    @it("it must return the correct number of patches")
    def test_get_patch_num(self):
        extractor = VitExtractor(model_name="dino_vitb8", device="cpu")
        shape = (1, 3, 224, 224)
        assert extractor.get_patch_num(shape) == 1 + (224 // 8) * (224 // 8)

    @it("it must return the correct head and embedding dimensions")
    def test_get_head_and_embedding_dim(self):
        small = VitExtractor(model_name="dino_vits8", device="cpu")
        base = VitExtractor(model_name="dino_vitb8", device="cpu")
        assert small.get_head_num() == 6
        assert base.get_head_num() == 12
        assert small.get_embedding_dim() == 384
        assert base.get_embedding_dim() == 768

    @it("it must extract features from input image")
    @mock.patch("torch.hub.load")
    def test_get_feature_from_input(self, mock_torchhub_load, dummy_img):
        # Setup: Create a fake model with .blocks containing dummy modules
        class DummyBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = mock.MagicMock()
                self.attn.attn_drop = mock.MagicMock()
                self.attn.qkv = mock.MagicMock()

            def forward(self, x):
                return x

        dummy_model = mock.MagicMock()
        dummy_model.eval = mock.MagicMock()
        dummy_model.blocks = [DummyBlock() for _ in range(12)]
        dummy_model.__call__ = mock.MagicMock()

        # Return dummy model from torch.hub.load
        mock_torchhub_load.return_value = dummy_model

        # Instantiate the extractor (loads dummy model)
        extractor = VitExtractor("dino_vitb8", device="cpu")

        # Simulate a hook adding something to outputs_dict
        dummy_output = torch.randn(1, 197, 384)
        extractor.outputs_dict[VitExtractor.BLOCK_KEY] = [dummy_output] * 12

        # Call the function under test
        features = extractor.get_feature_from_input(dummy_img)

        # Assertions
        assert isinstance(features, list)
        assert len(features) == 12
        for feature in features:
            assert isinstance(feature, torch.Tensor)
            assert feature.shape == (1, 197, 384)

    @it("it must get QKV features from input image")
    @mock.patch("gan.dino_struct.VitExtractor._register_hooks")
    @mock.patch("gan.dino_struct.VitExtractor._clear_hooks")
    @mock.patch("gan.dino_struct.VitExtractor._init_hooks_data")
    def test_get_qkv_feature_from_input(
        self, mock_init, mock_clear, mock_register, dummy_img
    ):

        extractor = VitExtractor(model_name="dino_vitb8", device="cpu")
        extractor.model = mock.MagicMock()

        dummy_qkv = [torch.randn(197, 3 * 12 * 64)]
        extractor.outputs_dict["qkv"] = dummy_qkv

        result = extractor.get_qkv_feature_from_input(dummy_img)

        mock_register.assert_called_once()
        extractor.model.assert_called_once_with(dummy_img)
        mock_clear.assert_called_once()
        mock_init.assert_called()
        assert result == dummy_qkv

    @it("it must get attention features from input image")
    @mock.patch("gan.dino_struct.VitExtractor._register_hooks")
    @mock.patch("gan.dino_struct.VitExtractor._clear_hooks")
    @mock.patch("gan.dino_struct.VitExtractor._init_hooks_data")
    def test_get_attn_feature_from_input(
        self, mock_init, mock_clear, mock_register, dummy_img
    ):

        extractor = VitExtractor(model_name="dino_vitb8", device="cpu")
        extractor.model = mock.MagicMock()

        dummy_attn = [
            torch.randn(1, 6, 197, 197)
        ]  # (batch, heads, patch_num, patch_num)
        extractor.outputs_dict["attn"] = dummy_attn

        result = extractor.get_attn_feature_from_input(dummy_img)

        mock_register.assert_called_once()
        extractor.model.assert_called_once_with(dummy_img)
        mock_clear.assert_called_once()
        mock_init.assert_called()
        assert result == dummy_attn

    @it("it must extract queries from QKV tensor")
    def test_get_queries_from_qkv(self):
        patch_num = 5
        head_num = 2
        embedding_dim = 8
        input_shape = (1, 3, 40, 40)

        qkv = (
            torch.arange(patch_num * 3 * embedding_dim)
            .float()
            .reshape(patch_num, 3 * embedding_dim)
        )

        extractor = VitExtractor(model_name="dino_vitb8", device="cpu")

        with mock.patch.object(
            extractor, "get_patch_num", return_value=patch_num
        ), mock.patch.object(
            extractor, "get_head_num", return_value=head_num
        ), mock.patch.object(
            extractor, "get_embedding_dim", return_value=embedding_dim
        ):

            queries = extractor.get_queries_from_qkv(qkv, input_shape)

            assert queries.shape == (head_num, patch_num, embedding_dim // head_num)

            expected_q = qkv.reshape(
                patch_num, 3, head_num, embedding_dim // head_num
            ).permute(1, 2, 0, 3)[0]
            assert torch.allclose(queries, expected_q)

    @it("it must extract keys from QKV tensor")
    @mock.patch.object(VitExtractor, "get_patch_num", return_value=6)
    @mock.patch.object(VitExtractor, "get_head_num", return_value=2)
    @mock.patch.object(VitExtractor, "get_embedding_dim", return_value=8)
    def test_get_keys_from_qkv(self, mock_dim, mock_head, mock_patch):
        patch_num = 6
        head_num = 2
        embedding_dim = 8
        input_shape = (1, 3, 48, 48)

        # Simulate a qkv tensor with shape [patch_num, 3 * embedding_dim]
        qkv = (
            torch.arange(patch_num * 3 * embedding_dim)
            .float()
            .reshape(patch_num, 3 * embedding_dim)
        )

        # Instantiate the extractor
        extractor = VitExtractor(model_name="dino_vitb8", device="cpu")

        # Call the method to get keys
        keys = extractor.get_keys_from_qkv(qkv, input_shape)

        # Check the shape: [head_num, patch_num, dim_per_head]
        assert keys.shape == (head_num, patch_num, embedding_dim // head_num)

        # Manually compute the expected key tensor for comparison
        expected_k = qkv.reshape(
            patch_num, 3, head_num, embedding_dim // head_num
        ).permute(1, 2, 0, 3)[1]
        assert torch.allclose(keys, expected_k)

    @it("it must extract values from QKV tensor")
    @mock.patch.object(VitExtractor, "get_patch_num", return_value=6)
    @mock.patch.object(VitExtractor, "get_head_num", return_value=2)
    @mock.patch.object(VitExtractor, "get_embedding_dim", return_value=8)
    def test_get_values_from_qkv(self, mock_dim, mock_head, mock_patch):
        patch_num = 6
        head_num = 2
        embedding_dim = 8
        input_shape = (1, 3, 48, 48)

        # Simulate a qkv tensor with shape [patch_num, 3 * embedding_dim]
        qkv = (
            torch.arange(patch_num * 3 * embedding_dim)
            .float()
            .reshape(patch_num, 3 * embedding_dim)
        )

        # Instantiate the extractor
        extractor = VitExtractor(model_name="dino_vitb8", device="cpu")

        # Call the method to get values
        values = extractor.get_values_from_qkv(qkv, input_shape)

        # Check the shape: [head_num, patch_num, dim_per_head]
        assert values.shape == (head_num, patch_num, embedding_dim // head_num)

        # Manually compute the expected value tensor for comparison
        expected_v = qkv.reshape(
            patch_num, 3, head_num, embedding_dim // head_num
        ).permute(1, 2, 0, 3)[2]
        assert torch.allclose(values, expected_v)

    @it("it must get keys from input image")
    @mock.patch.object(VitExtractor, "get_qkv_feature_from_input")
    @mock.patch.object(VitExtractor, "get_keys_from_qkv")
    def test_get_keys_from_input(
        self, mock_get_keys_from_qkv, mock_get_qkv_feature_from_input
    ):
        # Create dummy input image
        input_img = torch.randn(1, 3, 224, 224)
        layer_num = 11

        # Mock the qkv feature returned from the extractor
        fake_qkv = torch.randn(197, 3 * 768)  # [patches, 3 * embedding_dim]
        mock_get_qkv_feature_from_input.return_value = [None] * layer_num + [fake_qkv]

        # Mock the output of get_keys_from_qkv
        expected_keys = torch.randn(12, 197, 64)  # [heads, patches, dim_per_head]
        mock_get_keys_from_qkv.return_value = expected_keys

        # Create the extractor instance
        extractor = VitExtractor(model_name="dino_vitb8", device="cpu")

        # Call the method
        keys = extractor.get_keys_from_input(input_img, layer_num)

        # Check that methods were called correctly
        mock_get_qkv_feature_from_input.assert_called_once_with(input_img)
        mock_get_keys_from_qkv.assert_called_once_with(fake_qkv, input_img.shape)

        # Check that returned keys match mocked output
        assert torch.equal(keys, expected_keys)

    @it("it must compute self-similarity from input image keys")
    @mock.patch("gan.dino_struct.attn_cosine_sim")  # Patch the similarity function
    @mock.patch.object(VitExtractor, "get_keys_from_input")  # Patch internal method
    def test_get_keys_self_sim_from_input(self, mock_get_keys_from_input, mock_attn_cosine_sim):
        # Create dummy input image
        input_img = torch.randn(1, 3, 224, 224)
        layer_num = 11

        # Mocked keys tensor (heads, tokens, dim)
        fake_keys = torch.randn(12, 197, 64)
        mock_get_keys_from_input.return_value = fake_keys

        # Expected self-similarity map
        expected_ssim_map = torch.randn(1, 1, 197, 197)
        mock_attn_cosine_sim.return_value = expected_ssim_map

        # Create the extractor instance
        extractor = VitExtractor(model_name="dino_vitb8", device="cpu")

        # Call the method
        ssim_map = extractor.get_keys_self_sim_from_input(input_img, layer_num)

        # Ensure internal methods are called correctly
        mock_get_keys_from_input.assert_called_once_with(input_img, layer_num=layer_num)
        mock_attn_cosine_sim.assert_called_once()

        # Validate output
        assert torch.equal(ssim_map, expected_ssim_map)
