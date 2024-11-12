import os
from os import path
import torch
from torch import nn
from torchvision.utils import save_image

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from labels import reduced_labels, labels
from PIL import Image, ImageFile, ImageFilter
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import save_image
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import csv
import copy
import time
from torchvision import transforms
from tqdm import tqdm
import torch
import numpy as np
import os
import cv2
from sklearn.metrics import f1_score, roc_auc_score
import evaluate
from torch.cuda.amp import GradScaler, autocast
from scipy import signal
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():

    parser = argparse.ArgumentParser(
        description="Construct semantic segmentation using SAM"
    )

    parser.add_argument(
        "--in_train_dataset",
        type=str,
        required=True,
        help="path to input train dataset (synthetic or augmented images)",
    )

    parser.add_argument(
        "--in_test_dataset",
        type=str,
        required=True,
        help="path to input test dataset (real images)",
    )

    parser.add_argument(
        "--exp_folder",
        type=str,
        required=True,
        help="path to output folder of the experiment",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="training batch size",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=15,
        help="training batch size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="training epochs",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="training learning rate",
    )

    args = parser.parse_args()
    return args


def get_id(name):
    numbers = "0123456789"
    Id = [c for c in name if c in numbers]
    return int("".join(Id))


def sort_filenames(names):
    file_ids = [get_id(n) for n in names]
    Args = np.argsort(file_ids)
    return np.array(names)[Args]


class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        labels,
        transforms,
        shuffle=True,
        select_range=None,
        seed=0,
    ):
        """Build dataset class from the input folder

        Args:
            root (str): input dataset folder
            labels (list): labels list of the ids and colors in the segmentation map
            transforms (tuple): transformations to apply on images/segmentation maps
            shuffle (bool, optional): shuffle the dataset. Defaults to True.
            select_range (list, optional): restrict the training on a selected range of indexes. Defaults to None.
            seed (int, optional): fixed seed. Defaults to 0.
        """
        # fix the random seed
        np.random.seed(seed)
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = sort_filenames(os.listdir(os.path.join(root, "images")))
        self.masks = sort_filenames(os.listdir(os.path.join(root, "labels")))

        if len(self.masks) > len(self.imgs):
            self.masks = self.masks[: len(self.imgs)]

        # shuffle the dataset
        if shuffle:
            indexes = np.arange(len(self.masks))
            np.random.shuffle(indexes)
            self.imgs = self.imgs[indexes]
            self.masks = self.masks[indexes]
        # select a precise range of the dataset
        if select_range:
            self.imgs = self.imgs[select_range]
            self.masks = self.masks[select_range]
        self.reduced_labels = labels

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "labels", self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # reduce the labels map to the 11 classes we need if not already done when saving the labels
        mask = self.reduce_labels(mask)
        sample = {"image": image, "mask": mask}
        # apply necessary transformations
        if self.transforms:
            sample["image"] = self.transforms[0](sample["image"])
            sample["mask"] = self.transforms[1](sample["mask"])
        return sample

    def reduce_labels(self, cond_im):
        # Transform PIL image to np array
        modified_cond_im = np.copy(np.array(cond_im))
        # list of existing labels in the dataset
        labels = list(range(0, 34))
        # list of evaluated labels in our study
        eval_labels = [label.id for label in self.reduced_labels]
        for label in labels:
            # set the non evaluated to 0
            if not (label in eval_labels):
                modified_cond_im[modified_cond_im == label] = 0
            # replace the evaluated by its new label
            else:
                modified_cond_im[modified_cond_im == label] = eval_labels.index(label)
        # return to PIL image format
        cond_im = Image.fromarray(modified_cond_im)
        return cond_im

    def __len__(self):
        return len(self.masks)


def createDeepLabv3(outputchannels=len(reduced_labels), checkpoint=None):
    """Create Deeplabv3 model from pretrained checkpoint or from default weights

    Args:
        outputchannels (int, optional): number of output classes. Defaults to len(reduced_labels).
        checkpoint (str, optional): path to input checkpoint of a pretrained version. Defaults to None.

    Returns:
        nn.Module: segmentation model
    """
    # Load model from existing checkpoint
    if checkpoint != None:
        model = torch.load(checkpoint)
    # Load model from default weights
    else:
        model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        # Added a Sigmoid activation after the last convolution layer
        model.classifier = DeepLabHead(2048, outputchannels)
    # model = model.half()
    # Set the model in training mode
    model.train()
    return model


def log_IoU(exp_folder, step, results):
    """Log detailed mean IoU results on every class of the segmentation

    Args:
        exp_folder (str): path to experiment folder, where to save the results
        step (int): training step
        results (str): the segmentation results on each class
    """
    # Create log file if doesn't exist
    filename = os.path.join(exp_folder, "log_test_miou.txt")
    if os.path.isfile(filename):
        with open(filename, "r") as file:
            lines = file.readlines()
    else:
        lines = []
    # Save logs
    lines.append(f"step:{step};results:{results}\n")
    with open(filename, "w") as file:
        file.writelines(lines)


def log_loss(exp_folder, epoch, step, loss, mean_iou, phase):
    """log training loss and mean iou

    Args:
        exp_folder (str): path to experiment folder, where to save the results
        epoch (int): training epoch
        step (int): training step
        loss (float): training loss
        mean_iou (float): mean iou results
        phase (str): train or test phase
    """
    # create log files if not existing
    if phase == "Train":
        filename = os.path.join(exp_folder, "log_train.txt")
    else:
        filename = os.path.join(exp_folder, "log_test.txt")
    if os.path.isfile(filename):
        with open(filename, "r") as file:
            lines = file.readlines()
    else:
        lines = []
    # Save logs
    lines.append(f"epoch:{epoch};step:{step};loss:{loss};mean_IoU:{mean_iou}\n")
    with open(filename, "w") as file:
        file.writelines(lines)


def compute_metric(outputs, masks, metric, id2label):
    """compute the evaluation metric (mean IoU in our case) from model outputs

    Args:
        outputs (torch.Tensor): model outputs tensor
        masks (torch.Tensor): ground truth masks
        metric (evaluate_modules.metrics): evaluation metric
        id2label (dict): translation from segments ids to labels

    Returns:
        dict: evalutation metric on each class
    """
    # Interpolates the ouputs to get predicted classes
    logits_tensor = nn.functional.interpolate(
        outputs["out"],
        size=masks.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    # Offload the tensors to cpu
    y_pred = logits_tensor.cpu().numpy()
    y_true = masks.cpu().numpy()

    # Compute metrics
    metrics = metric.compute(
        predictions=y_pred,
        references=y_true,
        num_labels=len(id2label),
        ignore_index=0,
    )

    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    # Format results in dictionnaries
    metrics.update(
        {f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)}
    )
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

    return metrics


def cumulate(running_metric, batch_metric, keys):
    """Accumulate results to compute average performance per class on all the test dataset

    Args:
        running_metric (dict): previous batches results
        batch_metric (dict): actual batch mean iou metrics
        keys (list): dictionnary keys

    Returns:
        dict: accumulated performance
    """
    for key in keys:
        # Consider the accumulation if loss is not nan
        if not np.isnan(batch_metric[key]):
            # add metric
            if running_metric[key][1]:
                running_metric[key][0] += batch_metric[key]
            else:
                running_metric[key][0] = batch_metric[key]
            # increment count
            running_metric[key][1] += 1
    return running_metric


def reduce_metric(running_metric):
    """averaging of performances on all batches

    Args:
        running_metric (dict): sum of performance per class with the number of batches

    Returns:
        dict: average performance per class
    """
    results = {}
    mean = 0
    for key in running_metric:
        results[key] = running_metric[key][0] / running_metric[key][1]
        mean += results[key]
    results["iou_mean"] = mean / len(running_metric)
    return results


def eval_model(model, dataloaders, metric, criterion, id2label):
    """evaluate the model performances on test dataset

    Args:
        model (nn.Module): trained segmentation model
        dataloaders (dict): input dataloaders
        metric (evaluate_modules.metrics): evaluation metric
        criterion (loss criterion): _description_
        id2label (dict): dictionnary that maps classes ids to labels

    Returns:
        _type_: _description_
    """
    # init metrics dict and set model to eval mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    running_loss = 0
    step = 0
    running_metric = {
        "iou_road": [-1, 0],
        "iou_sidewalk": [-1, 0],
        "iou_building": [-1, 0],
        "iou_wall": [-1, 0],
        "iou_vegetation": [-1, 0],
        "iou_terrain": [-1, 0],
        "iou_sky": [-1, 0],
        "iou_person": [-1, 0],
        "iou_car": [-1, 0],
        "iou_truck": [-1, 0],
        "iou_bus": [-1, 0],
    }

    keys = list(running_metric.keys())
    for sample in tqdm(iter(dataloaders["Test"])):
        # load batches to device
        inputs = sample["image"].to(device, non_blocking=True)
        masks = (sample["mask"][:, 0, :, :].to(device, non_blocking=True) * 255).to(
            torch.long
        )

        with torch.set_grad_enabled(False):
            # compute outputs
            outputs = model(inputs)
            # compute loss and mean IoU
            loss = criterion(outputs["out"], masks)
            score_metrics = compute_metric(outputs, masks, metric, id2label)
            running_metric = cumulate(running_metric, score_metrics, keys)

            running_loss += loss.item()

        step += 1

    results = reduce_metric(running_metric)

    return running_loss / step, results["iou_mean"], results


def train_model(
    model,
    criterion,
    dataloaders,
    optimizer,
    exp_folder,
    num_epochs=3,
    batch_size=4,
    eval_steps=500,
    max_steps=10000,
):
    """train the model

    Args:
        model (nn.Module): segmentation model
        criterion (loss criterion): _description_
        dataloaders (dict): train and test dataloader
        optimizer (_type_): optimizing method for training
        exp_folder (str): path to output folder of the experiment
        num_epochs (int, optional): _description_. Defaults to 3.
        batch_size (int, optional): _description_. Defaults to 4.
        eval_steps (int, optional): _description_. Defaults to 500.
        max_steps (int, optional): _description_. Defaults to 10000.

    Returns:
        nn.Module: trained model
    """
    metric = evaluate.load("mean_iou")
    # id to label object
    id2label = {label.id: label.name for label in labels}

    # best_model_wts = copy.deepcopy(model.state_dict())  # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Init Gradient scaler
    scaler = GradScaler()

    model.train()

    best_performance = -1
    step = 0
    for epoch in range(1, num_epochs + 1):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary

        # Iterate over data.
        for sample in tqdm(iter(dataloaders["Train"])):
            inputs = sample["image"].to(device, non_blocking=True)
            masks = (sample["mask"][:, 0, :, :].to(device, non_blocking=True) * 255).to(
                torch.long
            )
            # zero the parameter gradients
            optimizer.zero_grad()

            # track history if only in train
            with torch.set_grad_enabled(True):
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs["out"], masks)
                score_metrics = compute_metric(outputs, masks, metric, id2label)
                log_loss(
                    exp_folder,
                    epoch,
                    step,
                    loss.item(),
                    score_metrics["mean_iou"],
                    "Train",
                )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            step += 1

            if not (step % eval_steps):
                eval_loss, eval_miou, results = eval_model(
                    model, dataloaders, metric, criterion, id2label, batch_size
                )
                log_loss(exp_folder, epoch, step, eval_loss, eval_miou, "Test")
                log_IoU(exp_folder, step, results)
                if eval_miou > best_performance:
                    best_performance = eval_miou
                    torch.save(model, exp_folder + "/" + "weights.pt")
            if max_steps and (step >= max_steps):
                break

    return model


def get_dataloader(train_dataset, test_dataset, batch_size, n_workers):
    """construct train and test dataloaders

    Args:
        train_dataset (str): batch size
        test_dataset (str):
        batch_size (int): batch size
        n_workers (int): number of dataloading workers

    Returns:
        dict: train and test dataloaders
    """
    data_transforms1 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((512, 1024)),
        ]
    )
    # segmentation maps must be resized with nearest interpolation to keep the int labels
    data_transforms2 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((512, 1024), interpolation=Image.NEAREST),
        ]
    )

    # Build datasets
    dataset = {
        "Train": CityscapesDataset(
            train_dataset,
            reduced_labels,
            (data_transforms1, data_transforms2),
            shuffle=True,
            seed=0,
        ),
        "Test": CityscapesDataset(
            test_dataset,
            reduced_labels,
            (data_transforms1, data_transforms2),
            shuffle=True,
            select_range=list(range(0, 600)),
            seed=0,
        ),
    }

    # Create dataloaders
    dataloader = {
        "Train": torch.utils.data.DataLoader(
            dataset["Train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
        ),
        "Test": torch.utils.data.DataLoader(
            dataset["Test"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
        ),
    }
    return dataloader


def main(args):
    """train segmentation model on train dataset and evaluate its performances of test dataset

    Args:
        args (argparse.Namespace): input arguments
    """

    # empty torch cache
    torch.cuda.empty_cache()

    # train and test dataloaders
    data_loader = get_dataloader(
        args.in_train_dataset, args.in_test_dataset, args.batch_size, args.num_workers
    )

    # construct model
    model = createDeepLabv3(
        outputchannels=len(labels),
        checkpoint=None,
    )

    # fix training loss function and optimization method
    criterion = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # create output folder if not existing
    exp_directory = args.exp_folder
    if not (os.path.isdir(exp_directory)):
        os.mkdir(exp_directory)

    # train segmentation model
    _ = train_model(
        model,
        criterion,
        data_loader,
        optimizer,
        exp_directory,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_steps=400,
        max_steps=50000,
    )


if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    # launch training/evaluation
    main(args)
