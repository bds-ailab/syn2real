import pytest
from unittest import mock
import numpy as np
import os
import torch
from PIL import Image
import random
import torch
from transformers import PretrainedConfig
from controlnet_sdxl.model import (
    import_model_class_from_model_name_or_path,
    save_model_card,
    encode_prompt,
)

it = pytest.mark.it
describe = pytest.mark.describe
skip = pytest.mark.skip


@describe("Test model import and saving functions")
class TestModelFunctions:

    @skip
    @it("should import the correct model class based on the model name or path")
    @mock.patch("transformers.PretrainedConfig.from_pretrained")
    def test_import_model_class_from_model_name_or_path(self, mock_from_pretrained):
        # Mocking the configuration return value
        mock_from_pretrained.return_value = PretrainedConfig(
            architectures=["CLIPTextModel"]
        )

        model_class = import_model_class_from_model_name_or_path("model/path", "main")
        assert model_class.__name__ == "CLIPTextModel"

        mock_from_pretrained.return_value = PretrainedConfig(
            architectures=["CLIPTextModelWithProjection"]
        )
        model_class = import_model_class_from_model_name_or_path("model/path", "main")
        assert model_class.__name__ == "CLIPTextModelWithProjection"

        mock_from_pretrained.return_value = PretrainedConfig(
            architectures=["UnsupportedModel"]
        )
        with pytest.raises(ValueError, match="UnsupportedModel is not supported."):
            import_model_class_from_model_name_or_path("model/path", "main")

    @skip
    @it("should save the model card correctly")
    @mock.patch("diffusers.utils.make_image_grid")
    @mock.patch("diffusers.utils.hub_utils.load_or_create_model_card")
    @mock.patch("diffusers.utils.hub_utils.populate_model_card")
    @mock.patch("PIL.Image.Image.save")
    @mock.patch("PIL.Image.Image.paste")
    def test_save_model_card(
        self,
        mock_paste,
        mock_save,
        mock_populate_model_card,
        mock_load_or_create_model_card,
        mock_make_image_grid,
    ):
        repo_id = "test_repo"
        base_model = "base_model"
        repo_folder = "repo_folder"
        image_mock = mock.Mock()
        image_mock.size = (10, 10)
        image_logs = [
            {
                "images": [image_mock, mock.Mock()],
                "validation_prompt": "A test prompt",
                "validation_image": image_mock,
            }
        ]

        # Create the repo_folder if it does not exist
        os.makedirs(repo_folder, exist_ok=True)

        # Call the function
        save_model_card(repo_id, image_logs, base_model, repo_folder)

        # Check if the README.md was saved correctly
        assert os.path.exists(os.path.join(repo_folder, "README.md"))
