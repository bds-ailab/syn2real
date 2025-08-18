import numpy as np
import pytest
import torch
from unittest import mock
from sklearn.metrics import confusion_matrix, classification_report
from baseline_model.eval_metrics_tools import conf_mtx, GradCamInspector
from baseline_model.config import REPORT_PATH


@pytest.mark.describe("Testing the conf_mtx function")
class TestConfMtx:
    @pytest.mark.it(
        "Must compute and return confusion matrix and classification report"
    )
    @mock.patch("matplotlib.pyplot.show")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("PIL.Image.open", return_value=mock.Mock())
    @mock.patch("matplotlib.pyplot.figure")
    @mock.patch("seaborn.heatmap")
    def test_conf_mtx(
        self, mock_open, mock_show, mock_image_open, mock_pyplot, mock_heatmap
    ):
        labels = np.array([0, 1, 2, 2, 1, 0])
        preds = np.array([0, 1, 1, 2, 1, 0])
        class_names = ["class_0", "class_1", "class_2"]

        conf_matrix, class_report_dict = conf_mtx(
            labels, preds, class_names, display=True, report=True
        )

        assert conf_matrix is not None
        assert class_report_dict is not None


@pytest.mark.describe("Testing the GradCamInspector class")
class TestGradCamInspector:

    @pytest.mark.it("Must compute gradients and activations correctly")
    @mock.patch("torch.no_grad", side_effect=torch.no_grad)
    @mock.patch("torch.mean")
    def test_compute_grad(self, mock_no_grad, mock_mean):
        model = mock.Mock()
        model.layer4 = [mock.Mock(), mock.Mock(), mock.Mock()]
        backward_hook_mock = mock.Mock()
        forward_hook_mock = mock.Mock()

        backward_hook_mock.remove.return_value = mock.Mock()
        forward_hook_mock.remove.return_value = mock.Mock()

        model.layer4[2].register_full_backward_hook.return_value = backward_hook_mock
        model.layer4[2].register_forward_hook.return_value = forward_hook_mock

        model.return_value = torch.rand((1, 10), requires_grad=True)

        inspector = GradCamInspector(model, {0: "class_0", 1: "class_1"})
        example_image = torch.zeros((1, 3, 224, 224))
        inspector.gradients = [torch.rand((1, 1024, 8, 8))]
        inspector.compute_grad(example_image)

        model.layer4[2].register_full_backward_hook.assert_called_once()
        model.layer4[2].register_forward_hook.assert_called_once()

        assert inspector.gradients is not None

    @pytest.mark.it("Must display GradCam map correctly")
    @mock.patch("matplotlib.pyplot.show")
    def test_display_gradcam(self, mock_show):
        model = mock.Mock()
        inspector = GradCamInspector(model, {0: "class_0", 1: "class_1"})
        example_image = torch.zeros((1, 3, 224, 224))

        # Simulate activations and pooled gradients
        inspector.activations = torch.rand((1, 1024, 8, 8))
        inspector.pooled_gradients = torch.rand(1024)
        inspector.pred_class = torch.tensor([0])

        inspector.display_gradcam(example_image)

        # Assertions for display
        mock_show.assert_called_once()
