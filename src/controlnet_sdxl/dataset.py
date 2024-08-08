import cv2
import numpy as np
import torch
import torch.utils.checkpoint
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from scipy import signal


def get_train_dataset(args, accelerator, logger):
    """Get the datasets: you can either provide your own training and evaluation files (see below)
    or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    Args:
        args (argparse.Namespace): Parser arguments
        accelerator (accelerate.Accelerator): Context manager that enables distributed training
        logger (logging.Logger): accelerate logger that can handle multi processing

    Raises:
        ValueError: if passed image column not found in dataset
        ValueError: if passed caption column not found in dataset
        ValueError: if passed conditioning image column not found in dataset

    Returns:
        Dataset: training dataset
    """

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir, cache_dir=args.cache_dir, trust_remote_code=True
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(
            f"conditioning image column defaulting to {conditioning_image_column}"
        )
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    with accelerator.main_process_first():
        # Shuffling dataset
        train_dataset = dataset["train"].shuffle(seed=args.seed)
        # Selecting maximum number of instances
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset


def prepare_train_dataset(args, dataset, accelerator):
    """Apply preparation transformations on input dataset

    Args:
        args (argparse.Namespace): Parser arguments
        dataset (Dataset): Input dataset
        accelerator (accelerate.Accelerator): Context manager that enables distributed training


    Returns:
        Dataset: transformed dataset
    """
    # Resize target images, transform them to tensors and normalize
    image_transforms = transforms.Compose(
        [
            transforms.Resize(
                (args.resolution // 2, args.resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            # transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # Resize conditioning images and transform them to tensors
    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(
                (args.resolution // 2, args.resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            # transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

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

    def augment_train(image, conditioning_image, syn_bool):
        """Add augmentation techniques on input image

        Args:
            image (pil image): Target image
            conditioning_image (pil image): Conditioning image
            syn_bool (bool): Synthetic or real ?

        Returns:
            pil image: Augmented conditioning image
        """
        if conditioning_image.size[0] != image.size[0]:
            conditioning_image = conditioning_image.resize(image.size)
        image_array = np.array(image)
        conditioning_image_array = np.array(conditioning_image)

        # Removing the segmentation maps for some images and leaving only canny edges for control
        # Will help the model understand the information from canny edges alone
        seg_removed = False
        # self.val variable will prevent applying augmentation ops during inference or validation
        if syn_bool:
            n = np.random.rand()
            # Applying this op on half of synthetic dataset since half of real dataset already
            # does not have segmentation maps (only canny)
            if n < 0.5:
                conditioning_image_array[:, :, :] = 0

        # Test if conditioning image does not have segmentation maps
        size = conditioning_image_array.shape
        if np.sum(conditioning_image_array == 0) >= 0.4 * (size[0] * size[1] * size[2]):
            seg_removed = True

        # Replacing some segments with black mask for the model to guess from shape
        # Will help the model generate plausible outputs from imperfect segmentations
        if not seg_removed:
            # Ids is a 1-channel representation of the segmentation maps where each segment has a
            # unique int id instead of an RGB color
            Ids = (
                conditioning_image_array[:, :, 0] * 2
                + (conditioning_image_array[:, :, 1] ** 1)
                + (
                    conditioning_image_array[:, :, 2] ** 3
                    + 0.00001
                    * conditioning_image_array[:, :, 0]
                    * conditioning_image_array[:, :, 1]
                    * conditioning_image_array[:, :, 2]
                )
            )
            Ids = np.uint8((255 * (Ids / np.max(Ids))).astype(int))
            Img_ids = np.zeros(conditioning_image_array.shape)
            for i in range(3):
                Img_ids[:, :, i] = Ids

            # removing some ids (segments) in random
            class_id = np.random.choice(np.unique(Ids), 3)
            for id in class_id:
                conditioning_image_array[Img_ids == id * np.ones(3)] = 0

        size = conditioning_image_array.shape
        if np.sum(conditioning_image_array == 0) >= 0.4 * (size[0] * size[1] * size[2]):
            seg_removed = True

        # Adding second control as Canny edges
        if args.canny_edges:
            n = np.random.rand()
            if n < 0.2 and (not seg_removed):
                # Not applying canny transformations for 20% of the dataset to leave just seg maps
                pass
            else:
                # Different thresholds for real in syn images because of the noise present in syn images
                if not syn_bool:
                    low_threshold = np.random.randint(0, 250)  # 50

                # Thresholds were selected experimentaly by looking for values that generates the same output
                # for both real and synthetic images
                else:
                    low_threshold = np.random.randint(100, 300)  # 150
                high_threshold = low_threshold + np.random.randint(20, 100)  # 120

                # Transforming the target image to canny edges
                canny_image = cv2.Canny(image_array, low_threshold, high_threshold)

                # Drawing thicker and more visible line with dilate
                kernel = np.ones((2, 2), np.uint8)
                canny_image = cv2.dilate(canny_image, kernel, iterations=1)
                canny_image = canny_image[:, :, None]

                # Formatting and normalizing the canny edges image to the same shape of source images
                canny_image = (
                    np.concatenate([canny_image, canny_image, canny_image], axis=2)
                    / 255
                )

                # Combining segmentation map with canny edges
                conditioning_image_array[canny_image == 1] = 255

        # Distort synthetic images
        # I observed that the model (after multiple rounds of training) started to distinguish
        # Segmentation maps of synthetic images only from their regularitym which makes it harder
        # to transform synthetic images to real in the inference, so I'm adding the distortions to
        # fool the model into thinking that synthetic images have distorted segmentation maps then
        # remove the irregularities in the inference to be more like a real images's seg map
        if syn_bool:
            # Generate a function to distort the image along x,y axis
            dist_y = generate_wave(conditioning_image_array.shape[0])
            dist_x = generate_wave(conditioning_image_array.shape[1])

            im_shifted = np.copy(conditioning_image_array)
            for i in range(conditioning_image_array.shape[1]):
                # Shift each column with 'shift' pixels, the value of the shift is given from the wave function
                shift = int(6 * dist_x[i])
                if shift > 0:
                    # Pad the missing pixels of the column after shifting
                    ndata = np.pad(im_shifted[:, i], ((shift, 0), (0, 0)), mode="edge")
                    im_shifted[:, i] = ndata[: len(ndata) - shift, :]
                elif shift < 0:
                    # Pad the missing pixels of the column after shifting
                    ndata = np.pad(im_shifted[:, i], ((0, -shift), (0, 0)), mode="edge")
                    im_shifted[:, i] = ndata[-shift:, :]

            for j in range(conditioning_image_array.shape[0]):
                # Shift each row with 'shift' pixels, the value of the shift is given from the wave function
                shift = int(6 * dist_y[j])
                if shift > 0:
                    # Pad the missing pixels of the row after shifting
                    ndata = np.pad(im_shifted[j, :], ((shift, 0), (0, 0)), mode="edge")
                    im_shifted[j, :] = ndata[: len(ndata) - shift, :]
                elif shift < 0:
                    # Pad the missing pixels of the row after shifting
                    ndata = np.pad(im_shifted[j, :], ((0, -shift), (0, 0)), mode="edge")
                    im_shifted[j, :] = ndata[-shift:, :]

            # Add random gaussian noise on top
            salt_noise = np.uint8(np.random.normal(0, 1, im_shifted.shape) < 0.4)
            im_shifted *= salt_noise
            conditioning_image = Image.fromarray(im_shifted)

        else:
            # Returning conditioning image in pil format
            conditioning_image = Image.fromarray(conditioning_image_array)
        return conditioning_image

    def preprocess_train(examples):
        """Apply augmentation techniques than transformations on images

        Args:
            examples (Dataset): a fragment of the input dataset

        Returns:
            Dataset: transformed version of the fragment
        """

        images = []
        conditioning_images = []
        for i in range(len(examples[args.image_column])):
            # convert image to RGB format
            im = examples[args.image_column][i].convert("RGB")
            cond_im = examples[args.conditioning_image_column][i].convert("RGB")

            # Augment conditioning images
            cond_im = augment_train(im, cond_im, examples["syn_or_real"][i])
            # Transform images
            images.append(image_transforms(im))
            conditioning_images.append(conditioning_image_transforms(cond_im))

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images

        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset


def collate_fn(examples):
    """stack inputs in torch tensors

    Args:
        examples (Dataset): input dataset fragment

    Returns:
        dict: dictionnary of stacked tensors
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack(
        [example["conditioning_pixel_values"] for example in examples]
    )
    conditioning_pixel_values = conditioning_pixel_values.to(
        memory_format=torch.contiguous_format
    ).float()

    prompt_ids = torch.stack(
        [torch.tensor(example["prompt_embeds"]) for example in examples]
    )

    add_text_embeds = torch.stack(
        [torch.tensor(example["text_embeds"]) for example in examples]
    )
    add_time_ids = torch.stack(
        [torch.tensor(example["time_ids"]) for example in examples]
    )
    syn_or_real = torch.stack(
        [torch.tensor(example["syn_or_real"]) for example in examples]
    )

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_ids": prompt_ids,
        "unet_added_conditions": {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        },
        "syn_or_real": syn_or_real,
    }
