import os
import pytest
from unittest import mock
import yaml
from auto_log_exp_mlflow import search_unlogged_exp_to_be_logged, log_exp
import mlflow as ml

it = pytest.mark.it
describe = pytest.mark.describe


@describe("Test the logging of the experiments on MLFlow")
class TestAutoLogOnMLFlow:

    # Mock data
    mock_exp_data = {
        "to_log": True,
        "logged": False,
        "exp_name": "test_experiment",
        "run_name": "test_run",
        "config_file": "/out/test_folder/config.yml",
    }

    mock_config_data = {"param1": "value1", "param2": "value2"}

    @it("Must correctly identify there is no unlogged experiments and return no path")
    @mock.patch("os.listdir")
    @mock.patch("os.path.isdir")
    @mock.patch(
        "builtins.open", new_callable=mock.mock_open, read_data=yaml.dump(mock_exp_data)
    )
    def test_search_unlogged_exp_to_be_logged_no_exp(
        self, mock_open, mock_isdir, mock_listdir
    ):
        # Mock the os functions
        mock_listdir.return_value = ["test_folder"]
        mock_isdir.return_value = True

        # Call the function
        unlogged_experiments, _ = search_unlogged_exp_to_be_logged()

        # Assertions
        assert len(unlogged_experiments) == 0

    @it("Must correctly identify unlogged experiments and return their paths")
    @mock.patch("os.listdir")
    @mock.patch("os.path.isdir")
    @mock.patch(
        "builtins.open", new_callable=mock.mock_open, read_data=yaml.dump(mock_exp_data)
    )
    def test_search_unlogged_exp_to_be_logged_with_exp(
        self, mock_open, mock_isdir, mock_listdir
    ):
        # Mock the os functions
        mock_listdir.side_effect = lambda path: (
            ["exp.yml", "config.yml"] if path == "/out/test_folder" else ["test_folder"]
        )
        mock_isdir.return_value = True

        # Call the function
        unlogged_experiments, exp_paths = search_unlogged_exp_to_be_logged()

        # Assertions
        assert len(unlogged_experiments) == 1
        assert unlogged_experiments[0]["exp_name"] == "test_experiment"
        assert exp_paths[0] == "/out/test_folder/exp.yml"

    @it("Must log experiments correctly and update the 'logged' flag")
    @mock.patch("mlflow.set_tracking_uri")
    @mock.patch("mlflow.set_experiment")
    @mock.patch("mlflow.start_run")
    @mock.patch("mlflow.log_artifact")
    @mock.patch("mlflow.log_param")
    @mock.patch("os.listdir")
    @mock.patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data=yaml.dump(mock_config_data),
    )
    def test_log_exp(
        self,
        mock_open,
        mock_listdir,
        mock_log_param,
        mock_log_artifact,
        mock_start_run,
        mock_set_experiment,
        mock_set_tracking_uri,
    ):
        # Mock the os functions
        mock_listdir.return_value = ["weights_file"]

        # Mock the mlflow functions
        mock_set_tracking_uri.return_value = None
        mock_set_experiment.return_value = None
        mock_start_run.return_value.__enter__.return_value = None
        mock_log_artifact.return_value = None
        mock_log_param.return_value = None

        # Call the function
        log_exp([self.mock_exp_data], ["/out/test_folder/exp.yml"])

        # Assertions
        mock_set_tracking_uri.assert_called_once_with("https://mlflow.sf.eviden.com/")
        mock_set_experiment.assert_called_once_with("test_experiment")
        mock_start_run.assert_called_once_with(run_name="test_run")
        mock_log_artifact.assert_called_once_with(
            "/models/test_experiment_test_run/weights_file"
        )
        mock_log_param.assert_any_call("param1", "value1")
        mock_log_param.assert_any_call("param2", "value2")

        # Check if the 'logged' flag is set to True
        assert self.mock_exp_data["logged"] is True
