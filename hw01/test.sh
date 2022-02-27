#!bin/bash

bash intent_train.sh --device=cuda:7 --num_workers=64 --num_epoch=300 --num_layer=2 --hidden_size=512 --lr=8.3e-7
bash intent_train.sh --device=cuda:7 --num_workers=64 --num_epoch=300 --num_layer=2 --hidden_size=1024 --lr=3.6e-2
bash intent_train.sh --device=cuda:7 --num_workers=64 --num_epoch=300 --num_layer=2 --hidden_size=2048 --lr=1e-2
