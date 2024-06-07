import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import torch
import seaborn as sns
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import PIL
from sklearn.metrics import confusion_matrix, classification_report
from config import DEVICE, CONF_PATH, REPORT_PATH


def conf_mtx(
    labels,
    preds,
    class_names,
    display=True,
    report=True,
    conf_matrix_path=CONF_PATH,
    class_report_path=REPORT_PATH,
):
    """function: display confusion matrix and performance evaluation report

    Args:
        labels (ndarray): actual labels of test instances
        preds (ndarray): predicted labels of test instances
        class_names (list): classes names for visualization
        display (bool, optional): conditional var to display confusion matrix. Defaults to True.
        report (bool, optional): conditional var to display report. Defaults to True.

    Returns:
        conf_matrix (ndarray): confusion matrix of size (num_classes)*(num_classes).
        class_report (str): evaluation report.
    """
    # Compute confusion matrix from predictions and actual labels
    conf_matrix = confusion_matrix(labels, preds)

    # Display results in heatmap format
    if display:
        plt.figure(figsize=(8, 6))
        # Plot confusion matrix with class names
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )

        # Add labels, title, and axes ticks
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.savefig(conf_matrix_path)
        plt.show()

    # Compute performance report with precision, recall, f1-score for each class and globaly
    class_report = classification_report(labels, preds, target_names=class_names)
    class_report_dict = classification_report(
        labels, preds, target_names=class_names, output_dict=True
    )

    # Display the report
    if report:
        print("Classification Report:")
        print(class_report)
        f = open(class_report_path, "a")
        f.write(class_report)
        f.close()

    return conf_matrix, class_report_dict


class GradCamInspector:
    def __init__(self, model, classes_id_names):
        """GradCam explanations for a given instance with a given model

        Args:
            model (nn.Module): Trained classification model (ResNet50 in this case)
            classes_id_names (dict): Classes names indexed by ids
        """
        self.model = model
        self.pooled_gradients = None
        self.pred_class = None
        self.gradients = None
        self.activations = None
        self.classes_id_names = classes_id_names

    def compute_grad(self, example_image, device=DEVICE):
        """function:

        Args:
            example_img (torch.Tensor): example instance to be explained
            device (str, optional): cuda or cpu. Defaults to torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
        """

        def backward_hook_gradcam(module, grad_input, grad_output):
            print("Backward hook running...")
            self.gradients = grad_output
            # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
            print(f"Gradients size: {self.gradients[0].size()}")
            # We need the 0 index because the tensor containing the gradients comes
            # inside a one element tuple.

        def forward_hook_gradcam(module, args, output):
            print("Forward hook running...")
            self.activations = output
            # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
            print(f"Activations size: {self.activations.size()}")

        # Attach hook to inspect this layer's gradient to the output class in backward run
        backward_hook = self.model.layer4[2].register_full_backward_hook(
            backward_hook_gradcam, prepend=False
        )

        # Attach hook to inspect this layer's output in forward run
        forward_hook = self.model.layer4[2].register_forward_hook(
            forward_hook_gradcam, prepend=False
        )

        # Run foward run to record outputs and take predictions
        pred = self.model(example_image.to(device))
        idx = pred.argmax(axis=1)

        # Run backward run to record gradients
        pred[:, idx[0]].backward()

        # Compute pooled gradients with respect to the output channels
        self.pooled_gradients = torch.mean(self.gradients[0], dim=[0, 2, 3])
        self.pred_class = idx

        # Removing the hooks
        backward_hook.remove()
        forward_hook.remove()

    def display_gradcam(self, example_img):
        """function: displays GradCam maps from computed gradient

        Args:
            example_img (torch.Tensor): example instance to be explained
        """

        # weight the channels by corresponding gradients
        for i in range(self.activations.size()[1]):
            self.activations[:, i, :, :] *= self.pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()

        # relu on top of the heatmap
        heatmap = F.relu(heatmap)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # Create a figure and plot the first image
        fig, ax = plt.subplots()
        ax.axis("off")  # removes the axis markers

        # First plot the original image
        ax.imshow(to_pil_image(example_img[0], mode="RGB"))

        # Resize the heatmap to the same size as the input image and defines
        # a resample algorithm for increasing image resolution
        # we need heatmap.detach() because it can't be converted to numpy array while
        # requiring gradients
        overlay = to_pil_image(heatmap.detach(), mode="F").resize(
            (224, 224), resample=PIL.Image.BICUBIC
        )

        # Apply any colormap you want
        cmap = colormaps["jet"]
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

        # Plot the heatmap on the same axes,
        # but with alpha < 1 (this defines the transparency of the heatmap)
        ax.imshow(overlay, alpha=0.3, interpolation="nearest")
        ax.set_title(
            f"GradCam map explanation \n Prediced label: {self.classes_id_names[self.pred_class.cpu().numpy()[0]]}"
        )
        # Show the plot
        plt.show()
