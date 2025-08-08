import pytest
from unittest import mock
import json
import numpy as np
import os
from segment_anything import SamAutomaticMaskGenerator
from controlnet_sdxl.segment import (
    read_json_data,
    write_json_data,
    construct_seg_model,
    generate_mask,
    merge_masks,
    main,
)


@pytest.mark.describe("TestSegmentationFunctions")
class TestSegmentationFunctions:

    @pytest.mark.it("should read json data correctly from a json file")
    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data=json.dumps([{"conditioning_image": "path/to/image.png"}]),
    )
    def test_read_json_data(self, mock_open):
        result = read_json_data("test.json")
        assert result == [{"conditioning_image": "path/to/image.png"}]

    @pytest.mark.it("should write json data correctly to a json file")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_write_json_data(self, mock_open):
        data = [{"conditioning_image": "path/to/image.png"}]
        write_json_data("output.json", data)
        mock_open.assert_called_with("output.json", "w")
        mock_open().write.assert_called()

    @pytest.mark.it("should construct the segmentation model correctly")
    @mock.patch("huggingface_hub.hf_hub_download", return_value="mock_path")
    @mock.patch("segment_anything.build_sam_vit_h", return_value=mock.Mock())
    def test_construct_seg_model(self, mock_hf_hub_download, mock_build_sam_vit_h):
        mask_generator = construct_seg_model()
        assert isinstance(mask_generator, SamAutomaticMaskGenerator)

    @pytest.mark.it("should generate masks for a given image")
    @mock.patch("cv2.imread")
    @mock.patch("cv2.cvtColor")
    def test_generate_mask(self, mock_cvtColor, mock_imread):
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        mock_cvtColor.return_value = mock_image
        mock_mask_generator = mock.Mock()
        mock_mask_generator.generate.return_value = [
            {"segmentation": np.random.randint(0, 2, (100, 100), dtype=bool)}
        ]

        masks = generate_mask("path/to/image.png", mock_mask_generator)
        assert len(masks) == 1
        assert "segmentation" in masks[0]

    @pytest.mark.it("should merge the predicted masks correctly")
    def test_merge_masks(self):
        masks = [
            {"segmentation": np.random.randint(0, 2, (100, 100), dtype=bool)}
            for _ in range(5)
        ]
        merged_mask = merge_masks(masks)
        assert merged_mask.shape == (100, 100, 3)

    @pytest.mark.it("should run the main function without errors")
    @mock.patch("os.path.isdir", return_value=False)
    @mock.patch("os.mkdir")
    @mock.patch(
        "controlnet_sdxl.segment.read_json_data",
        return_value=[{"conditioning_image": "path/to/image.png"}],
    )
    @mock.patch("controlnet_sdxl.segment.construct_seg_model", return_value=mock.Mock())
    @mock.patch(
        "controlnet_sdxl.segment.generate_mask",
        return_value=[
            {"segmentation": np.random.randint(0, 2, (100, 100), dtype=bool)}
        ],
    )
    @mock.patch(
        "controlnet_sdxl.segment.merge_masks",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    @mock.patch("cv2.imwrite")
    def test_main(
        self,
        mock_write,
        mock_merge,
        mock_generate,
        mock_construct,
        mock_read,
        mock_mkdir,
        mock_isdir,
    ):
        mock_args = mock.Mock()
        mock_args.in_dataset = "input.json"
        mock_args.out_dataset = "output_folder"
        mock_args.start_idx = 0
        mock_args.end_idx = 0

        main(mock_args)

        # Check if the output folder was created
        mock_mkdir.assert_called_once()
        # Check if the correct number of images were processed
        assert mock_write.call_count == 1
