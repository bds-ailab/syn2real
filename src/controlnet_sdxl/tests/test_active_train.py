import pytest
import subprocess
from unittest import mock
from controlnet_sdxl.active_train import (
    read_train_config,
    controlnet_train,
    unet_train,
    prepare_data,
    active_train_loop,
)  # Adjust the import according to your actual module name

it = pytest.mark.it
describe = pytest.mark.describe


@describe("read_train_config function")
class TestReadTrainConfig:

    @it("should read configuration parameters from config.yml")
    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data='{"nb_rounds": 2, "round0": {"in_json": "input0.json", "out_json": "output0.json", "unet_path": "unet0.pth", "controlnet_path": "controlnet0.pth", "output_dir": "output_dir"}, "round1": {"in_json": "input1.json", "out_json": "output1.json", "unet_path": "unet1.pth", "controlnet_path": "controlnet1.pth", "output_dir": "output_dir2"}}',
    )
    def test_read_train_config(self, mock_open):
        config = read_train_config()
        assert config["nb_rounds"] == 2
        assert config["round0"]["in_json"] == "input0.json"


@describe("controlnet_train function")
class TestControlNetTrain:

    @it("should execute the training command")
    @mock.patch("subprocess.Popen")
    def test_controlnet_train(self, mock_popen):
        mock_popen.return_value.wait.return_value = (
            0  # Simulate successful command execution
        )
        result = controlnet_train(
            "base_model_dir",
            "unet_dir",
            "controlnet_dir",
            "output_dir",
            1e-4,
            8,
            1000,
            1,
        )
        assert result == 0
        mock_popen.assert_called_once()  # Ensure the command was called


@describe("unet_train function")
class TestUNetTrain:

    @it("should execute the training command")
    @mock.patch("subprocess.Popen")
    def test_unet_train(self, mock_popen):
        mock_popen.return_value.wait.return_value = (
            0  # Simulate successful command execution
        )
        result = unet_train(
            "base_model_dir",
            "unet_dir",
            "controlnet_dir",
            "output_dir",
            1e-4,
            8,
            1000,
            1,
        )
        assert result == 0
        mock_popen.assert_called_once()  # Ensure the command was called


@describe("prepare_data function")
class TestPrepareData:

    @it("should execute the data preparation command")
    @mock.patch("subprocess.Popen")
    def test_prepare_data(self, mock_popen):
        mock_popen.return_value.wait.return_value = (
            0  # Simulate successful command execution
        )
        result = prepare_data(
            "input.json", "output.json", "unet_path", "controlnet_path"
        )
        assert result == 0
        mock_popen.assert_called_once()  # Ensure the command was called


@describe("active_train_loop function")
class TestActiveTrainLoop:

    @it("should execute training rounds based on configuration")
    @mock.patch("os.mkdir")
    @mock.patch(
        "pathlib.Path.is_dir", return_value=False
    )  # Simulate that directories do not exist
    @mock.patch("controlnet_sdxl.active_train.controlnet_train")
    @mock.patch("controlnet_sdxl.active_train.unet_train")
    @mock.patch("controlnet_sdxl.active_train.prepare_data")
    def test_active_train_loop(
        self,
        mock_prepare_data,
        mock_unet_train,
        mock_controlnet_train,
        mock_mkdir,
        mock_is_dir,
    ):
        config = {
            "nb_rounds": 2,
            "round0": {
                "in_json": "input0.json",
                "out_json": "output0.json",
                "unet_path": "unet0.pth",
                "unet_dir2": "unet_dir2_0",
                "controlnet_path": "controlnet0.pth",
                "controlnet_dir2": "controlnet_dir2_0",
                "output_dir": "output_dir0",
                "output_dir2": "output_dir2_0",
                "base_model_path": "base_model_dir",
                "base_model_dir2": "base_model_dir2_0",
                "lr": 1e-4,
                "lr2": 1e-4,
                "bs": 8,
                "bs2": 8,
                "max_steps": 1000,
                "max_steps2": 1000,
                "acc_steps": 1,
                "acc_steps2": 1,
            },
            "round1": {
                "in_json": "input1.json",
                "out_json": "output1.json",
                "unet_path": "unet1.pth",
                "unet_dir2": "unet_dir2_1",
                "controlnet_path": "controlnet1.pth",
                "controlnet_dir2": "controlnet_dir2_0",
                "output_dir": "output_dir1",
                "output_dir2": "output_dir2_1",
                "base_model_path": "base_model_dir",
                "base_model_dir2": "base_model_dir2_0",
                "lr": 1e-4,
                "lr2": 1e-4,
                "bs": 8,
                "bs2": 8,
                "max_steps": 1000,
                "max_steps2": 1000,
                "acc_steps": 1,
                "acc_steps2": 1,
            },
        }

        active_train_loop(config)

        assert mock_prepare_data.call_count == 1  # Ensure prepare_data was called once
        assert (
            mock_controlnet_train.call_count == 2
        )  # Ensure controlnet_train was called for each round
        assert (
            mock_unet_train.call_count == 2
        )  # Ensure unet_train was called for each round
