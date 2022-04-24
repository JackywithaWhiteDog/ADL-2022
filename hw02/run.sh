#!bin/bash

export CUDA_VISIBLE_DEVICES="4"

# "${1}": path to the context file.
# "${2}": path to the testing file.
# "${3}": path to the output predictions.

python ./test.py \
  --multi_model ./ckpt/multiple_choice \
  --qa_model ./ckpt/qa \
  --per_device_eval_batch_size 32 \
  --multi_max_seq_length 512 \
  --qa_max_seq_length 384 \
  --max_answer_length 150 \
  --output_dir . \
  --overwrite_output \
  --cache_dir ./cache \
  --test_file "${2}" \
  --context_file "${1}" \
  --output_file "${3}"

# python ./test.py \
#   --multi_model ./ckpt/multiple_choice/roberta/checkpoint-4071 \
#   --qa_model ./ckpt/qa/roberta_new/checkpoint-38240 \
#   --per_device_eval_batch_size 32 \
#   --multi_max_seq_length 512 \
#   --qa_max_seq_length 384 \
#   --max_answer_length 150 \
#   --output_dir . \
#   --overwrite_output \
#   --cache_dir ./cache \
#   --test_file ./data/valid.json \
#   --context_file ./data/context.json \
#   --output_file ./pred/valid_prediction.csv

# python ./test.py \
#   --multi_model ./ckpt/multiple_choice/albert-tiny/checkpoint-4071 \
#   --qa_model ./ckpt/qa/roberta_new/checkpoint-38240 \
#   --per_device_eval_batch_size 32 \
#   --multi_max_seq_length 512 \
#   --qa_max_seq_length 384 \
#   --max_answer_length 150 \
#   --output_dir . \
#   --overwrite_output \
#   --cache_dir ./cache \
#   --test_file ./data/valid.json \
#   --context_file ./data/context.json \
#   --output_file ./pred/valid_prediction.csv

# python ./test.py \
#   --multi_model ./ckpt/multiple_choice/roberta_no_pretrained/checkpoint-4071 \
#   --qa_model ./ckpt/qa/roberta_new/checkpoint-38240 \
#   --per_device_eval_batch_size 32 \
#   --multi_max_seq_length 512 \
#   --qa_max_seq_length 384 \
#   --max_answer_length 150 \
#   --output_dir . \
#   --overwrite_output \
#   --cache_dir ./cache \
#   --test_file ./data/valid.json \
#   --context_file ./data/context.json \
#   --output_file ./pred/valid_prediction.csv

# python ./test.py \
#   --multi_model ./ckpt/multiple_choice/roberta/checkpoint-4071 \
#   --qa_model ./ckpt/qa/roberta_512/checkpoint-27680 \
#   --per_device_eval_batch_size 32 \
#   --multi_max_seq_length 512 \
#   --qa_max_seq_length 384 \
#   --max_answer_length 150 \
#   --output_dir . \
#   --overwrite_output \
#   --cache_dir ./cache \
#   --test_file ./data/test.json \
#   --context_file ./data/context.json \
#   --output_file ./pred/test_prediction_new.csv
