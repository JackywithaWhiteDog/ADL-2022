#!bin/bash

export CUDA_VISIBLE_DEVICES="4"

python3.8 intent.py \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --logging_steps 250 \
  --evaluation_strategy steps \
  --eval_steps 250 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --cache_dir ./cache \
  --train_file ./data/intent/train.json \
  --validation_file ./data/intent/eval.json \
  --test_file ./data/intent/test.json \
  --output_dir ./ckpt/intent


