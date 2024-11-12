#!/bin/bash

# This shell script will use the given model checkpoints to infer a transformed dataset from the synthetic one and save it in the ouput folder (will create the folder if doesn't exist as well as the json file)
python /sdxl/prepare_new_dataset.py --in_json_file="/data/cityscape/syn_city/data_complete_captionned.jsonl" --out_json_file="/out/inference_test/train.jsonl" --out_dataset_folder="/out/controlnet_sdxl_active_round2_distortion/checkpoint-20000/controlnet/" --controlnet_path="/out/controlnet_bdd/round1_controlnet/checkpoint-20000/controlnet/" --unet_path="/out/controlnet_sdxl_unlocked_attention/unet/"
