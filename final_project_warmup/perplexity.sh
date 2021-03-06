#!bin/bash

DEVICE=5

DATA_TYPE=in_domain
PRE_TRAINED=anlg

# DIALOGUES=./models/${DATA_TYPE}/grf-${DATA_TYPE}_${PRE_TRAINED}_eval/dialogues.txt

# DIALOGUES=./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75/generated_dialogues.txt
DIALOGUES=./ckpt/t5/out_of_domain/wd1e-1_lr5e-3_batch32_source50_target75/generated_dialogues.txt

CUDA_VISIBLE_DEVICES=${DEVICE} \
python3.8 ./perplexity.py $DIALOGUES
