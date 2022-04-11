#!bin/bash

export CUDA_VISIBLE_DEVICES="4"

python ./test.py \
  --multi_model ./ckpt/multiple_choice/sample/checkpoint-4000 \
  --qa_model ./ckpt/qa/roberta/checkpoint-30000 \
  --per_device_eval_batch_size 16 \
  --multi_max_seq_length 512 \
  --qa_max_seq_length 384 \
  --max_answer_length 150 \
  --output_dir . \
  --overwrite_output \
  --cache_dir ./cache \
  --test_file ./data/test.json \
  --context_file ./data/context.json \
  --output_file ./pred/test_prediction.csv

# python ./test.py \
#   --multi_model ./ckpt/multiple_choice/sample/checkpoint-4000 \
#   --qa_model ./ckpt/qa/roberta/checkpoint-30000 \
#   --per_device_eval_batch_size 16 \
#   --multi_max_seq_length 512 \
#   --qa_max_seq_length 384 \
#   --max_answer_length 150 \
#   --output_dir . \
#   --overwrite_output \
#   --cache_dir ./cache \
#   --test_file ./data/valid.json \
#   --context_file ./data/context.json \
#   --output_file ./pred/valid_prediction.csv
