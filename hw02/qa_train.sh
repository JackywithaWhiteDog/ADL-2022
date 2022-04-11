#!bin/bash

export CUDA_VISIBLE_DEVICES="7"

python qa_train.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 8 \
  --max_seq_length 384 \
  --max_answer_length 150 \
  --logging_steps 1000 \
  --evaluation_strategy steps \
  --eval_steps 3000 \
  --save_steps 10000 \
  --output_dir ./tmp/qa_roberta_batch8_lr2_384 \
  --overwrite_output \
  --cache_dir ./cache \
  --train_file ./data/train.json \
  --validation_file ./data/valid.json \
  --context_file ./data/context.json
