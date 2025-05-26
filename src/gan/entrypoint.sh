#!/bin/bash

python -c "print('Hello world, script is working!')"
python -c "import wandb; wandb.login(key='$WANDB_KEY')"
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('$HF_TOKEN')"

/gan/launch.sh
