import torch
import torch.nn as nn
import torch.optim as optim

# Paths to train and val datasets (syn and real)
TRAIN_PATH = "/data/syn2real/train/"
VAL_PATH = "/data/syn2real/validation/"

# Path to save trained model weights
MODEL_PATH = "/models/resnet50_latest.pth"

# Split rate to split train dataset to train and test
SR = 0.15

# Training batch size
BATCH_SIZE = 32
# Number of workers for batches loading
NUM_WORKERS = 2

# Computing device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# File to save the loss/epoch logs
LOG_FILE = "/output/log.txt"

# Number of training epochs
NUM_EPOCHS = 1
# Learning rate
LR = 0.001
# Optimization method
OPTIMIZER = optim.Adam
# Loss function to optimize
CRITEREON = nn.CrossEntropyLoss()

# Path to save confusion matrix image
CONF_PATH = "/output/confusion_matrix.png"
# Path to save classification report
REPORT_PATH = "/output/classification_report.txt"
