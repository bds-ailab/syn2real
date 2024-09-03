import torch

import argparse
import os
import pandas as pd
import sys

sys.path.append("/apps/lib")
import numpy as np
import pprint


# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
# details.
_SIGMA = 10
# The following is used to make the metric more human readable. See the paper
# for more details.
_SCALE = 1000


def mmd(x, y):
    """
    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(
            -gamma
            * (
                -2 * torch.matmul(x, x.T)
                + torch.unsqueeze(x_sqnorms, 1)
                + torch.unsqueeze(x_sqnorms, 0)
            )
        )
    )
    k_xy = torch.mean(
        torch.exp(
            -gamma
            * (
                -2 * torch.matmul(x, y.T)
                + torch.unsqueeze(x_sqnorms, 1)
                + torch.unsqueeze(y_sqnorms, 0)
            )
        )
    )
    k_yy = torch.mean(
        torch.exp(
            -gamma
            * (
                -2 * torch.matmul(y, y.T)
                + torch.unsqueeze(y_sqnorms, 1)
                + torch.unsqueeze(y_sqnorms, 0)
            )
        )
    )

    return _SCALE * (k_xx + k_yy - 2 * k_xy)


if __name__ == "__main__":
    path = "/out/CLIP"
    embeddings = np.load(path + "/embeddings.npy")
    labels = np.load(path + "/labels.npy")
    dist_real_aug = mmd(embeddings[labels == 0], embeddings[labels == 1])
    dist_real_syn = mmd(embeddings[labels == 2], embeddings[labels == 1])

    print(
        f"MMD distance between synthetic and real datasets (CLIP embeddings) : ",
        dist_real_syn,
    )
    print(
        f"MMD distance between augmented and real datasets (CLIP embeddings) : ",
        dist_real_aug,
    )
