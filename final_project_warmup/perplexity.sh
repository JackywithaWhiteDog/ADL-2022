#!bin/bash

export CUDA_VISIBLE_DEVICES="7"

python3.8 ./perplexity.py \
    ./ckpt/in_domain/wd2_lr2e-3_batch32/generated_dialogues.txt
