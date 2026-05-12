#!/bin/bash

sudo apt-get update
python3.11 -m pip install --user --upgrade pip setuptools wheel
python3.11 -m pip install --user --upgrade vllm-tpu "huggingface_hub[hf_transfer]"

~/.local/bin/vllm --version

# Start vllm.
# ~/.local/bin/vllm serve "Qwen/Qwen3-4B"   
#     --download_dir /tmp
#     --tensor_parallel_size=1
#     --max-model-len=8192
#     --gpu-memory-utilization=0.95
