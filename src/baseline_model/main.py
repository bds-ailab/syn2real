from data_tools import split_train_val, preprocess_resnet50
from model_tools import load_resnet50, train, evaluate_model
from eval_metrics_tools import conf_mtx, GradCamInspector
import torch
import matplotlib.pyplot as plt
import argparse
from time import time
from config import TRAIN_PATH, VAL_PATH, MODEL_PATH, SR, NUM_EPOCHS


def benchmark():

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run the benchmarking process by training the baseline model on synthetic datasets and evaluating him on real datasets"
    )

    # Add arguments
    parser.add_argument(
        "--train_path",
        type=str,
        default=TRAIN_PATH,
        help="Path to train (synthetic) dataset",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=VAL_PATH,
        help="Path to validation (real) dataset",
    )
    parser.add_argument(
        "--model_path", type=str, default=MODEL_PATH, help="Path to save model weights"
    )

    # Parse the arguments
    args = parser.parse_args()

    train_images_dict, val_images_dict, report = split_train_val(args.train_path, SR)
    train_dataloader = preprocess_resnet50(train_images_dict)
    val_dataloader = preprocess_resnet50(val_images_dict)
    print("Data loaded successfully")

    model = load_resnet50(report["num_classes"])
    print("Model loaded successfully")

    num_batches = len(train_dataloader)
    train(model, train_dataloader, NUM_EPOCHS, num_batches, report)
    torch.save(model.state_dict(), MODEL_PATH)

    val_preds, val_labels = evaluate_model(model, val_dataloader)
    print("==================")
    print("Evaluation results on validation instances (syn data)")
    val_conf_matrix, val_report = conf_mtx(
        val_labels, val_preds, report["class_names"], display=True, report=True
    )

    test_images_dict, _, test_data_report = split_train_val(args.val_path, 0)
    test_dataloader = preprocess_resnet50(test_images_dict)
    test_preds, test_labels = evaluate_model(model, test_dataloader)
    print("==================")
    print("Evaluation results on test instances (real data)")
    test_conf_matrix, test_report = conf_mtx(
        test_labels, test_preds, report["class_names"], display=True, report=True
    )


if __name__ == "__main__":
    benchmark()
