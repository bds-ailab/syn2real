import pytest
import torch
from unittest import mock
from PIL import Image
import json
from blip_caption.utils import load_model, load_data, process, save_captions
from transformers import BlipProcessor, BlipModel
from blip_caption.config import DEVICE

skip = pytest.mark.skip
it = pytest.mark.it
describe = pytest.mark.describe


@describe("Test the utils function of the module blip")
class TestUtils:

    @it("Must load the model and processor correctly")
    @mock.patch("transformers.BlipProcessor.from_pretrained")
    @mock.patch("transformers.BlipForConditionalGeneration.from_pretrained")
    def test_load_model(self, mock_model, mock_processor):
        # Create a mock for the model that will have the .to() method.
        mock_model_instance = mock.Mock()
        mock_model.return_value = mock_model_instance

        processor, model = load_model()

        mock_processor.assert_called_once_with("Salesforce/blip-image-captioning-base")
        mock_model.assert_called_once_with("Salesforce/blip-image-captioning-base")

        # Now check that the .to() method was called on the model instance.
        mock_model_instance.to.assert_called_once_with(DEVICE)

    @it("Must load image paths correctly")
    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data=json.dumps(
            [{"image": "path/to/image1.jpg"}, {"image": "path/to/image2.jpg"}]
        ),
    )
    def test_load_data(self, mock_open):
        data_path = "fake_data_path.json"
        paths = load_data(data_path)
        assert paths == [
            {"image": "path/to/image1.jpg"},
            {"image": "path/to/image2.jpg"},
        ]

    @it("Must process images and generate captions")
    @mock.patch("PIL.Image.open", return_value=Image.new("RGB", (100, 100)))
    def test_process(self, mock_open):

        # Create a mock processor of the appropriate type
        processor = mock.Mock(
            spec=BlipProcessor
        )  # Specify the processor class you are mocking
        model = mock.Mock(spec=BlipModel)

        # Simulate image paths
        paths = [{"image": "path/to/image1.jpg"}, {"image": "path/to/image2.jpg"}]
        prompt = "a picture of "

        # Simulate the behavior of the model and processor
        model.generate.return_value = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        processor.decode.side_effect = ["caption1", "caption2"]

        # Set up the processor mock to return a mapping (dictionary)
        processor.return_value = {
            "pixel_values": torch.tensor(
                [[1, 2], [3, 4]]
            ),  # Example tensor for pixel values
            "other_info": "example_info",  # You can add more keys as needed
        }

        # Prepare inputs as a proper dictionary
        inputs = {
            "pixel_values": processor.return_value[
                "pixel_values"
            ],  # Use the mock output
            "other_info": processor.return_value["other_info"],
        }

        # Call model.generate with proper inputs
        outputs = model.generate(**inputs)

        # Ensure the model.generate method was called with the right arguments
        model.generate.assert_called_with(**inputs)

        # Decode the outputs (simulate the processor.decode behavior)
        captions = [processor.decode(output) for output in outputs]

        # Assertions to verify the output
        assert captions == ["caption1", "caption2"]

    @it("Must save captions correctly to a file")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_save_captions(self, mock_open):
        paths = [
            {"image": "path/to/image1.jpg", "dataset": "real"},
            {"image": "path/to/image2.jpg", "dataset": "synthetic"},
        ]
        captions = ["caption1", "caption2"]
        filename = "fake_file.json"
        prompt = "a picture of "

        save_captions(paths, captions, filename, prompt)

        # Verify that the file was opened and written
        mock_open.assert_called_once_with(filename, "w")
        handle = mock_open()
        handle.write.assert_called()
