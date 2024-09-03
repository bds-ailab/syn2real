from datasets import load_dataset
import json
from labels import labels, reduced_labels
from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments
import torch
from torch import nn
import evaluate
from transformers import Trainer
import numpy as np
from PIL import Image
from scipy import signal
import cv2
from PIL import ImageFilter, ImageEnhance, ImageFile
import yaml
import datasets


ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_train_config():
    """Read configuration parameters from config.yml file in current directory

    Returns:
        dict: Dictionnary of configuration parameters
    """
    config_path = "config.yml"
    # Open yml file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def preprocess_train(examples):
    """Preprocess input images

    Args:
        examples (Dataset): a fragment of the input dataset

    Returns:
        Dataset: preprocessed version of the fragment
    """

    images = []
    conditioning_images = []
    for i in range(len(examples["pixel_values"])):
        # convert image to RGB format
        im = examples["pixel_values"][i].convert("RGB")
        cond_im = examples["label"][i].convert("RGB")

        # Transform images
        images.append(im)
        conditioning_images.append(cond_im)

    examples["pixel_values"] = images
    examples["label"] = conditioning_images

    return examples


def reduce_labels(cond_im):
    """Reduce the labels in the segmentation maps by setting the non evaluated categories to 0

    Args:
        cond_im (PIL image): input segmentation map (in labels format not colors)

    Returns:
        PIL image: segmentation map after labels reduction
    """
    # Transform PIL image to np array
    modified_cond_im = np.copy(np.array(cond_im))
    # list of existing labels in the dataset
    labels = list(range(0, 34))
    # list of evaluated labels in our study
    eval_labels = [label.id for label in reduced_labels]
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


def augment(input_image, input_label):
    """Apply basic data augmentation techniques to input images/labels

    Args:
        input_image (PIL image): input image to augment
        input_label (PIL image): input segmentation map to augment

    Returns:
        (PIL image, PIL image): augmented image, augmented labels
    """

    example_image = np.copy(
        np.array(input_image.resize(input_label.size).convert("RGB"))
    )
    example_label = np.copy(np.array(input_label))

    n = np.random.rand()
    if n < 0.5:
        # Randomly deleting some labels in the image
        label_ids = np.unique(example_label)
        id2delete = np.random.choice(label_ids, 4)
        for label in id2delete:
            indexes = example_label == label
            example_image[indexes] = 0
    else:
        # Randomly adding black square patches in the image
        # to force the model to predict the segmentation only from context
        imin, jmin = np.random.randint(
            0, example_label.shape[0] - 200
        ), np.random.randint(0, example_label.shape[1] - 200)
        imax, jmax = np.random.randint(imin + 50, imin + 200), np.random.randint(
            jmin + 50, jmin + 200
        )
        example_image[imin:imax, jmin:jmax, :] = 0

    def generate_wave(size):
        """Generates random wave function to distort segmentation maps images

        Args:
            size (int): signal lenght

        Returns:
            numpy.ndarray: array of wave values
        """
        # Random frequencies
        freq1 = 0.1 * np.random.rand()
        freq2 = 0.01 * np.random.rand()
        freq3 = 0.001 * np.random.rand()
        # Random magnitudes
        amp1 = np.random.rand()
        amp2 = np.random.rand()

        distortion = np.zeros(size)
        for i in range(len(distortion)):
            # Generate wave values
            distortion[i] = (
                np.random.rand() * amp1 * np.sin(freq1 * i) * np.cos(freq2 * i)
                + amp2 * np.sin(freq3 * i)
                + ((0.00001 * i) ** 5)
            )

        # Smooth the function
        distortion = signal.medfilt(distortion, 3)
        # Normalize the values beween 0-1
        distortion /= distortion.max()
        return distortion

    # Randomly distorting the images/labels, this augmentation technique should be
    # applied on the pairs images labels
    n = np.random.rand()
    if n < 0.3:

        # Generate a function to distort the image along x,y axis
        dist_y = generate_wave(example_label.shape[0])
        dist_x = generate_wave(example_label.shape[1])

        im_shifted_label = np.copy(example_label)
        im_shifted_image = np.copy(example_image)

        for i in range(example_label.shape[1]):
            # Shift each column with 'shift' pixels, the value of the shift is given from the wave function
            shift = int(6 * dist_x[i])
            if shift > 0:
                # Pad the missing pixels of the column after shifting
                ndata = np.pad(im_shifted_label[:, i], ((shift, 0),), mode="edge")
                im_shifted_label[:, i] = ndata[: len(ndata) - shift]
                ndata = np.pad(
                    im_shifted_image[:, i], ((shift, 0), (0, 0)), mode="edge"
                )
                im_shifted_image[:, i] = ndata[: len(ndata) - shift, :]
            elif shift < 0:
                # Pad the missing pixels of the column after shifting
                ndata = np.pad(im_shifted_label[:, i], ((0, -shift),), mode="edge")
                im_shifted_label[:, i] = ndata[-shift:]
                ndata = np.pad(
                    im_shifted_image[:, i], ((0, -shift), (0, 0)), mode="edge"
                )
                im_shifted_image[:, i] = ndata[-shift:, :]

        for j in range(example_label.shape[0]):
            # Shift each row with 'shift' pixels, the value of the shift is given from the wave function
            shift = int(6 * dist_y[j])
            if shift > 0:
                # Pad the missing pixels of the row after shifting
                ndata = np.pad(
                    im_shifted_image[j, :], ((shift, 0), (0, 0)), mode="edge"
                )
                im_shifted_image[j, :] = ndata[: len(ndata) - shift, :]
                ndata = np.pad(im_shifted_label[j, :], ((shift, 0),), mode="edge")
                im_shifted_label[j, :] = ndata[: len(ndata) - shift]
            elif shift < 0:
                # Pad the missing pixels of the row after shifting
                ndata = np.pad(
                    im_shifted_image[j, :], ((0, -shift), (0, 0)), mode="edge"
                )
                im_shifted_image[j, :] = ndata[-shift:, :]
                ndata = np.pad(im_shifted_label[j, :], ((0, -shift),), mode="edge")
                im_shifted_label[j, :] = ndata[-shift:]

        # Randomly bluring the images
        n = np.random.rand()
        if n < 0.4:
            input_image = Image.fromarray(im_shifted_image).filter(ImageFilter.BLUR)
        else:
            input_image = Image.fromarray(im_shifted_image)
        input_label = Image.fromarray(im_shifted_label)
    else:
        # Randomly bluring the images
        n = np.random.rand()
        if n < 0.4:
            input_image = Image.fromarray(example_image).filter(ImageFilter.BLUR)
        else:
            input_image = Image.fromarray(example_image)
        input_label = Image.fromarray(example_label)
    return input_image, input_label


def main(config):

    torch.cuda.empty_cache()

    # Reading the synthetic train dataset
    syn_dataset = load_dataset(config["train_dataset"], trust_remote_code=True)["train"]
    # Renaming a dataset column
    syn_dataset = syn_dataset.rename_column("image", "pixel_values")

    # shuffling the dataset
    train_ds = syn_dataset.shuffle(seed=1)

    # Reading the real evaluation dataset
    real_dataset = load_dataset(config["test_dataset"], trust_remote_code=True)["train"]

    # Renaming a dataset column
    real_dataset = real_dataset.rename_column("image", "pixel_values")
    # shuffling the dataset and selecting a portion
    val_ds = real_dataset.shuffle(seed=1).select(range(1000))

    # Preprocessing images
    train_ds = train_ds.with_transform(preprocess_train)
    val_ds = val_ds.with_transform(preprocess_train)

    # id to label object
    id2label = {label.id: label.name for label in labels}
    label2id = {label.name: label.id for label in labels}

    # Segformer Preprocessor
    processor = SegformerImageProcessor()
    # color jitter for colors augmentation
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    def train_transforms(example_batch):
        """applies the augmentation techniques of a batch of images and preprocess the batch

        Args:
            example_batch (dict): batch of images/labels

        Returns:
            dict: batch of augmented and preprocessed images/labels
        """
        images = []
        labels = []
        for i in range(len(example_batch["pixel_values"])):
            # Color jittering for the input images
            example_image = jitter(example_batch["pixel_values"][i])
            # reduce labels
            example_label = reduce_labels(example_batch["label"][i])
            # augment images/labels
            aug_image, aug_label = augment(
                example_image,
                example_label,
            )
            images.append(aug_image)
            labels.append(aug_label)
        # Segformer preprocess
        inputs = processor(images, labels)
        return inputs

    def val_transforms(example_batch):
        """preprocess the batch. (no augmentation since this is for the validation dataset)

        Args:
            example_batch (dict): batch of images/labels

        Returns:
            dict: batch of preprocessed images/labels
        """
        images = [x for x in example_batch["pixel_values"]]
        labels = [reduce_labels(x) for x in example_batch["label"]]
        inputs = processor(images, labels)
        return inputs

    # Set transforms
    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)

    # Load the model from a previous checkpoint or from the pretrained hub version
    if config["pretrained_model"] == "None":
        pretrained_model_name = "nvidia/mit-b0"
    else:
        pretrained_model_name = config["pretrained_model"]

    # Load model
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name, id2label=id2label, label2id=label2id
    )

    # Read training hyperparameters from config yaml
    epochs = config["epochs"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    # Define whole training strategy
    training_args = TrainingArguments(
        config["save_model_repo"],
        dataloader_num_workers=config["dataloader_num_workers"],
        dataloader_persistent_workers=True,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=20,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        logging_steps=1,
        eval_accumulation_steps=4,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    # Define evaluation metric
    # Mean IoU = Intersection over Union
    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        """Compute mean IoU metric from the model predictions

        Args:
            eval_pred (tuple): model output logits and GT labels

        Returns:
            _type_: _description_
        """
        with torch.no_grad():
            # read model outputs and GT labels
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            # scale the logits to the size of the label
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            # Compute predicted labels
            pred_labels = logits_tensor.detach().cpu().numpy()
            # Compute mean IoU
            metrics = metric.compute(
                predictions=pred_labels,
                references=labels,
                num_labels=len(id2label),
                ignore_index=0,
                reduce_labels=processor.do_reduce_labels,
            )

            # add per category metrics as individual key-value pairs
            per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
            per_category_iou = metrics.pop("per_category_iou").tolist()

            # Format results in dictionnaries
            metrics.update(
                {
                    f"accuracy_{id2label[i]}": v
                    for i, v in enumerate(per_category_accuracy)
                }
            )
            metrics.update(
                {f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)}
            )

            return metrics

    # Define model trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    # Train model !
    trainer.train()


if __name__ == "__main__":
    # Read yaml config file
    config = read_train_config()
    # Run the training
    main(config)
