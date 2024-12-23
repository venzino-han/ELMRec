# !/bin/bash
python3 seq_reranker.py --cuda --data_dir data/beauty/
python3 seq_reranker.py --cuda --data_dir data/sports/
python3 seq_reranker.py --cuda --data_dir data/toys/
