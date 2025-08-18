import numpy as np
from IPython.display import clear_output
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast
from baseline_model.config import DEVICE, LOG_FILE, LR, CRITEREON, OPTIMIZER


def load_resnet50(num_classes, device=DEVICE):
    """function: load pretrained ResNet50 weights and replace the fc layer

    Args:
        num_classes (int): number of classes for output layer
        device (str): cuda or cpu

    Returns:
        model (torchvision.models): ResNet model ready to be finetuned
    """
    # Load the pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    # Adapting the last classification layer with the number of classes in our dataset
    model.fc = nn.Linear(num_features, num_classes)

    # Attaching model to cuda or cpu
    model.to(device)

    return model


def save_log(loss, epoch, num_epochs, log_file=LOG_FILE):
    with open(log_file, "a") as f:
        f.writelines([f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}\n"])


def train(
    model,
    train_dataloader,
    num_epochs,
    num_batches,
    report,
    optimizer_method=OPTIMIZER,
    criterion=CRITEREON,
    lr=LR,
    device=DEVICE,
    log_file=LOG_FILE,
):
    """function: train the ResNet model on given dataloader

    Args:
        model (torchvision.models): ResNet model ready to be finetuned
        train_dataloader (Dataloader): Training dataloader
        num_epochs (int): number of training epochs
        num_batches (int): number of batches in trainloader
        report (dict): informations report from previous pipeline steps
        optimizer_method (method, optional): optimization method for gradient descent. Defaults to optim.Adam.
        criterion (method, optional): loss function to be optimized. Defaults to nn.CrossEntropyLoss().
        lr (float, optional): learning rate. Defaults to 0.001.
        device (str, optional): computing device. Defaults to torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
    """

    # Init training optimizer
    optimizer = optimizer_method(model.parameters(), lr=lr)

    # Init Gradient scaler
    scaler = GradScaler()

    # Init loss history register
    loss_history = []

    # Iterate on epochs number
    for epoch in range(num_epochs):
        # Setting the model to training mode
        model.train()
        # Init running loss over each epoch
        running_loss = 0.0
        b = 0
        # Optimization for each batch
        for inputs, labels in train_dataloader:
            b += 1
            clear_output(wait=True)
            # Attaching batch data to device cuda or cpu
            # Note: non blocking is used to optimize the waiting time for batch transfe to GPU
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Using autocast as context manager to allow running in mixed precision
            with autocast():
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass and optimize
            # Using scaler to rescale back the updated parameters
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Saving loss results
            running_loss += loss.item()
            loss_history.append(loss)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Batch [{b}/{num_batches}], Loss: {loss:.4f}"
            )

        # Saving the training logs
        save_log(running_loss / num_batches, epoch, num_epochs)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/num_batches:.4f}")

    report["loss_history"] = loss_history
    print("Finished Training")


def evaluate_model(model, dataloader, device=DEVICE):
    """function: model predictions inference

    Args:
        model (nn.Module): Trained model to evaluate
        dataloader (Dataloader): Validation dataloader
        device (str, optional): computing device. Defaults to DEVICE.

    Returns:
        (ndarray, ndarray): (model predictions, actual labels).
    """
    # Setting the model to evaluation mode
    model.eval()
    all_preds = []
    all_labels = []
    # Disable gradient calculations for memory optim
    with torch.no_grad():
        # Iterating on eval batches
        for inputs, labels in dataloader:

            # Transfer data to GPU
            # Note: non blocking is used to optimize the waiting time for batch transfer to GPU
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            # Predictions inference
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Saving data
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)
