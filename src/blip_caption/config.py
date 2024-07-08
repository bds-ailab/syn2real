import torch


# Computing device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset absolute path
DATA_PATH = "/data/cityscape/data.json"

# Output saving file for captions/paths
FILE_NAME = "/data/cityscape/data_captionned.json"

# Prompt: Question to ask BLIP about the picture
PROMPT = "a picture of "
