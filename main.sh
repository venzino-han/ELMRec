# !/bin/bash
model_version=2
batch_size=4
# python3 pretrain.py --cuda --data_dir data/beauty/ --model_version $model_version --batch_size $batch_size
# python3 seq_reranker.py --cuda --data_dir data/beauty/ --model_version $model_version --batch_size $batch_size

# python3 pretrain.py --cuda --data_dir data/toys/ --model_version $model_version --batch_size $batch_size
python3 seq_reranker.py --cuda --data_dir data/toys/ --model_version $model_version --batch_size $batch_size

# python3 pretrain.py --cuda --data_dir data/sports/ --model_version $model_version --batch_size $batch_size
# python3 seq_reranker.py --cuda --data_dir data/sports/ --model_version $model_version --batch_size $batch_size
