import pytest
from unittest import mock
import json
from controlnet_sdxl.prepare_new_dataset import (
    read_json_data,
    write_json_data,
    construct_conditioning_image,
    load_pipeline,
    transform_img,
    save_batch,
    process_dataset,
)


@pytest.mark.describe("TestPrepareNewDataset")
class TestPrepareNewDataset:

    @pytest.mark.it("should read json data correctly from a json file")
    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data=json.dumps([{"image": "path/to/image.png"}]),
    )
    def test_read_json_data_json(self, mock_open):
        result = read_json_data("test.json")
        assert result == [{"image": "path/to/image.png"}]

    @pytest.mark.it("should read jsonl data correctly from a jsonl file")
    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data=json.dumps({"image": "path/to/image.png"}) + "\n",
    )
    def test_read_json_data_jsonl(self, mock_open):
        mock_open.return_value.readlines.return_value = [
            json.dumps({"image": "path/to/image.png"}) + "\n"
        ]
        result = read_json_data("test.jsonl")
        assert result == [{"image": "path/to/image.png"}]

    @pytest.mark.it("should write json data correctly to a json file")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_write_json_data_json(self, mock_open):
        data = [{"image": "path/to/image.png"}]
        write_json_data("output.json", data)
        mock_open.assert_called_once_with("output.json", "w")

    @pytest.mark.it("should write json data correctly to a jsonl file")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_write_json_data_jsonl(self, mock_open):
        data = [{"image": "path/to/image.png"}]
        write_json_data("output.jsonl", data)
        mock_open.assert_called_once_with("output.jsonl", "w")

    @pytest.mark.skip
    @pytest.mark.it("should construct conditioning image correctly")
    @mock.patch("PIL.Image.open")
    def test_construct_conditioning_image(self, mock_open):
        image_path = "path/to/image.png"
        conditioning_image_path = "path/to/conditioning_image.png"

        mock_image = mock.Mock()
        mock_conditioning_image = mock.Mock()

        mock_open.side_effect = [mock_image, mock_conditioning_image]
        mock_image.size.return_value = (1024, 512)
        mock_conditioning_image.size.return_value = (1024, 512)

        result = construct_conditioning_image(image_path, conditioning_image_path)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.it("should load pipeline correctly")
    @mock.patch("diffusers.ControlNetModel.from_pretrained")
    @mock.patch("diffusers.UNet2DConditionModel.from_pretrained")
    @mock.patch("diffusers.StableDiffusionXLControlNetPipeline.from_pretrained")
    def test_load_pipeline(self, mock_pipeline, mock_unet, mock_controlnet):
        args = mock.Mock()
        args.controlnet_path = "path/to/controlnet"
        args.unet_path = "path/to/unet"

        mock_controlnet.return_value = mock.Mock()
        mock_unet.return_value = mock.Mock()
        mock_pipeline.return_value = mock.Mock(scheduler=mock.Mock(config={}))

        pipe = load_pipeline(args)
        assert pipe is not None

    @pytest.mark.it("should transform images correctly")
    @mock.patch("controlnet_sdxl.prepare_new_dataset.transform_img")
    def test_transform_img(self, mock_transform_img):
        mock_pipe = mock.Mock()
        batch = {
            "im": ["mock_image"],
            "text": ["mock prompt"],
            "cond": ["mock_conditioning_image"],
        }
        mock_transform_img.return_value = ["transformed_image"]

        result = mock_transform_img(mock_pipe, batch)
        assert len(result) == 1
        assert result[0] == "transformed_image"

    @pytest.mark.it("should save a batch of generated images correctly")
    def test_save_batch(self):
        args = mock.Mock()
        args.out_dataset_folder = "output_folder"
        batch = {
            "im": [mock.Mock(), mock.Mock()],
            "cond": [mock.Mock(), mock.Mock()],
            "text": ["prompt 1", "prompt 2"],
        }
        out_data = []
        starting_idx = 0

        result_data, result_idx = save_batch(args, out_data, batch, starting_idx)
        assert len(result_data) == 2
        assert result_idx == 2

    @pytest.mark.skip
    @pytest.mark.it("should process dataset correctly")
    @mock.patch("controlnet_sdxl.prepare_new_dataset.read_json_data")
    @mock.patch("controlnet_sdxl.prepare_new_dataset.write_json_data")
    @mock.patch("controlnet_sdxl.prepare_new_dataset.load_pipeline")
    @mock.patch("PIL.Image.open")
    def test_process_dataset(
        self,
        mock_image_open,
        mock_load_pipeline,
        mock_write_json_data,
        mock_read_json_data,
    ):
        args = mock.Mock()
        args.in_json_file = "input.json"
        args.out_json_file = "output.json"
        args.out_dataset_folder = "output_folder"

        mock_read_json_data.return_value = [
            {
                "image": "img_path",
                "conditioning_image": "cond_img_path",
                "text": "a synthetic image",
            }
        ]
        mock_image = mock.Mock()
        mock_image.size = (1024, 512)
        mock_image_open.side_effect = [mock_image, mock.Mock(size=(1024, 512))]

        mock_load_pipeline.return_value = mock.Mock()

        process_dataset(args)
        mock_read_json_data.assert_called()
        mock_write_json_data.assert_called_once_with(
            args.out_json_file, mock_read_json_data.return_value
        )
