#!/bin/bash
export TRAIN_DATA_DIR="/data/cityscapes_aug"
export TEST_DATA_DIR="/data/cityscapes_real"
export EXP_OUT_FOLDER="/out/train_segmenter_test"

python train_deeplab.py --in_train_dataset=$TRAIN_DATA_DIR --in_test_dataset=$TEST_DATA_DIR --exp_folder=$EXP_OUT_FOLDER --batch_size=4 --num_workers=15 --epochs=12 --learning_rate=1e-5