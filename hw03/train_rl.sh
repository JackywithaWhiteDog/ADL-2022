#!bin/bash

export CUDA_VISIBLE_DEVICES="6"

python3.8 ./main.py \
    --do_train \
    --rl \
    --model_name_or_path ./ckpt/mt5 \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --max_source_length 256 \
    --max_target_length 64 \
    --text_column maintext \
    --summary_column title \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --source_prefix "summarize: " \
    --output_dir ./ckpt/mt5_rl
