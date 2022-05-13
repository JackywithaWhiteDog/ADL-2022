#!bin/bash

export CUDA_VISIBLE_DEVICES="7"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

python3.8 $SCRIPT_DIR/perplexity.py \
    ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75/generated_dialogues.txt
