import torch

# Computing device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset absolute path
DATA_PATH = "/data/mock_data/fill50k/target/"

# Output saving file for captions/paths
FILE_NAME = "/data/mock_data/fill50k/generated_captions.txt"

# Special token to add at the beginning of the captions
TOKEN = "a picture of "
