# !/bin/bash
model_version=0
batch_size=64
# python3 pretrain.py --cuda --data_dir data/merged/ --model_version $model_version --batch_size $batch_size
python3 seq_reranker.py --cuda --data_dir data/merged/ --model_version $model_version --batch_size $batch_size
