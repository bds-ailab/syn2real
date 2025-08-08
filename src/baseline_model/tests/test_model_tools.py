import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest import mock
from torch.utils.data import DataLoader
from baseline_model.model_tools import load_resnet50, save_log, train, evaluate_model

it = pytest.mark.it
describe = pytest.mark.describe


class SimpleModel(nn.Module):
    """
    Simple class to mimick a basic model allowing us to test the model_tools functions
    """

    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(224 * 224 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


@describe("Testing the tool functions around the ResNet50 training")
class TestModelTools:
    # Mock data
    mock_data = [(torch.zeros(3, 224, 224), 0)]
    mock_train_dataloader = DataLoader(mock_data)
    mock_val_dataloader = DataLoader(mock_data)
    mock_report = {}

    @it("Must load ResNet50 and replace the final layer correctly")
    @mock.patch("torchvision.models.resnet50")
    def test_load_resnet50(self, mock_resnet50):
        mock_model = mock.Mock()
        mock_model.fc = nn.Linear(2048, 1000)
        mock_model.fc.in_features = 2048
        mock_resnet50.return_value = mock_model
        model = load_resnet50(num_classes=10, device="cpu")

        # Assertions
        assert model.fc.out_features == 10

    @it("Must write the correct log to the file")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_save_log(self, mock_open):
        save_log(loss=0.1234, epoch=0, num_epochs=10, log_file="mock_log.txt")

        # Assertions
        mock_open.assert_called_once_with("mock_log.txt", "a")
        mock_open().writelines.assert_called_once_with(["Epoch [1/10], Loss: 0.1234\n"])

    @it("Must train the model and update the loss history")
    @mock.patch("torch.cuda.amp.GradScaler")
    @mock.patch("torch.optim.Adam")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_train(self, mock_adam, mock_grad_scaler, mock_open):

        model = SimpleModel(num_classes=10)
        mock_adam.return_value = optim.Adam(model.parameters())
        mock_grad_scaler.return_value = mock.Mock()
        self.mock_report = {}
        train(
            model=model,
            train_dataloader=self.mock_train_dataloader,
            num_epochs=1,
            num_batches=1,
            report=self.mock_report,
            optimizer_method=optim.Adam,
            criterion=nn.CrossEntropyLoss(),
            lr=0.001,
            device="cpu",
            log_file="mock_log.txt",
        )

        # Assertions
        assert "loss_history" in self.mock_report
        assert len(self.mock_report["loss_history"]) > 0

    @it("Must evaluate the model and return predictions and labels")
    @mock.patch("torch.no_grad", side_effect=torch.no_grad)
    def test_evaluate_model(self, mock_no_grad):
        model = SimpleModel(num_classes=10)
        with torch.no_grad():
            preds, labels = evaluate_model(
                model, self.mock_val_dataloader, device="cpu"
            )

        # Assertions
        assert isinstance(preds, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert len(preds) == len(labels)
