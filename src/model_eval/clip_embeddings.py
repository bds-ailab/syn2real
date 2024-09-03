from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
from os import path
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch


def extract_embeddings(images_list, out_path):
    """extracts the images embeddings in the semantic space using CLIP model

    Args:
        images_list (list): list of images paths
        out_path (str): output saving folder

    Returns:
        list: embeddings list
    """

    # Empty cuda cache memory
    torch.cuda.empty_cache()

    # Load CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    # Set model to evaluation mode
    model.eval()

    # Define the images processor
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    batch_size = 4
    images_embeddings = []
    for i in range(0, 10, batch_size):
        # Load images
        imgs = [
            Image.open(name).resize((1024, 512)).convert("RGB")
            for name in images_list[i : i + batch_size]
        ]

        # Preprocess the images
        inputs = processor(images=imgs, return_tensors="pt")

        # Extract the features
        image_features = np.array(model.get_image_features(**inputs).detach().cpu())

        # Collect the features
        for j in range(len(imgs)):
            images_embeddings.append(image_features[j])
        print(f"{i}/{len(images_list)}")

    images_embeddings = np.array(images_embeddings)
    # Save the features embeddings
    np.save(out_path + "embeddings.npy", images_embeddings)
    return images_embeddings


def project_embeddings(embeddings, labels, method, out_path):
    """Project the extracted embeddings using PCA or UMAP

    Args:
        embeddings (ndarray): embeddings array of shape (n_images, dim_embeddings)
        labels (list): labels of each embedding (generated=0, real=1 or synthetic=2)
        method (str): projection method 'pca' or 'umap'
        out_path (str): saving folder path
    """
    # Set the projection method
    if method == "pca":
        projector = PCA(n_components=2)
    elif method == "umap":
        projector = umap.UMAP(n_neighbors=4, random_state=42)

    # Fit the projection method & transform the embeddings
    projector.fit(embeddings)
    trans = projector.transform(embeddings)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        trans[labels == 0, 0],
        trans[labels == 0, 1],
        s=3,
        alpha=0.5,
        c="green",
        label="augmented",
    )
    plt.scatter(
        trans[labels == 1, 0],
        trans[labels == 1, 1],
        s=3,
        alpha=0.5,
        c="green",
        label="real",
    )
    plt.scatter(
        trans[labels == 2, 0],
        trans[labels == 2, 1],
        s=3,
        alpha=0.5,
        c="red",
        label="synthetic",
    )

    plt.legend()
    plt.title(
        f"{method} projection of CLIP semantic embeddings of Generated images",
        fontsize=12,
    )
    plt.savefig(out_path + "projection.png")


def pca_variance(embeddings, out_path):
    """extracts PCA explained variance analysis

    Args:
        embeddings (ndarray): embeddings array of shape (n_images, dim_embeddings)
        out_path (str): saving folder path
    """

    # Fit the PCA
    pca = PCA(n_components=10).fit(embeddings)
    # Extract the explained variance
    exp_var_pca = pca.explained_variance_ratio_

    # Cumulative sum of eigenvalues; This will be used to create step plot
    # for visualizing the variance explained by each principal component.

    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    # Create the visualization plot
    plt.figure(figsize=(8, 6))
    plt.bar(
        range(0, len(exp_var_pca)),
        exp_var_pca,
        alpha=0.5,
        align="center",
        label="Individual explained variance",
    )
    plt.step(
        range(0, len(cum_sum_eigenvalues)),
        cum_sum_eigenvalues,
        where="mid",
        label="Cumulative explained variance",
    )
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal component index")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path + "pca_exp_var.png")


if __name__ == "__main__":

    # Paths for the three datasets
    images_path1 = "/data/cityscapes_aug/images/"
    images_list1 = os.listdir(images_path1)
    images_path2 = "/data/cityscapes_real/images/"
    images_list2 = os.listdir(images_path2)
    images_path3 = "/data/cityscapes_syn/images/"
    images_list3 = os.listdir(images_path3)

    # Defining labels
    labels = [0] * len(images_list1) + [1] * len(images_list2) + [2] * len(images_list3)

    # Saving path
    out_path = "./"
    np.save(out_path + "labels.npy", labels)

    images_list = (
        [images_path1 + pth for pth in images_list1]
        + [images_path2 + pth for pth in images_list2]
        + [images_path3 + pth for pth in images_list3]
    )

    # Extract embeddings
    embeddings = extract_embeddings(images_list, out_path)
    # Project and plot embeddings
    project_embeddings(embeddings, labels, "pca", out_path)
    # PCA analysis
    pca_variance(embeddings, out_path)
