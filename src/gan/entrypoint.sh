#!/bin/bash

python -c "print('Hello world, script is working!')"
python -c "import wandb; wandb.login(key='3ca3b5a6911f69877ed6f769370ad36ce15d2ae6')"
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"

/gan/launch.sh