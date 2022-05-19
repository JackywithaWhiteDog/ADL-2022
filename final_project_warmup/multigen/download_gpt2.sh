#!bin/bash

mkdir -p models
cd models
mkdir -p gpt2-small
cd gpt2-small
wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin
wget -O gpt2-vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json
