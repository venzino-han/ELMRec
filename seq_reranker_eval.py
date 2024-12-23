import os
import torch
import random
import argparse
from tqdm import tqdm
from transformers import T5Tokenizer
from util.utils import SeqDataLoader, SeqBatchify, now_time, evaluate_ndcg, evaluate_hr


parser = argparse.ArgumentParser(description='ELMRec')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory for loading the data')
parser.add_argument('--model_version', type=int, default=0,
                    help='1: t5-base; 2: t5-large; 3: t5-3b; 4: t5-11b; otherwise: t5-small')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--checkpoint', type=str, default='./ELMRec/',
                    help='directory to load the final model')
parser.add_argument('--num_beams', type=int, default=30,
                    help='number of beams')
parser.add_argument('--top_n', type=int, default=10,
                    help='number of items to predict')
parser.add_argument('--num_test_samples', type=int, default=40000,)

# both tasks : {'Sports': 10, 'Beauty': 15, 'Toys': 10}
parser.add_argument('--N', type=int, default=10,
                    help='number of additional candidates')
args = parser.parse_args()

if args.model_version == 1:
    model_version = 't5-base'
elif args.model_version == 2:
    model_version = 't5-large'
elif args.model_version == 3:
    model_version = 't5-3b'
elif args.model_version == 4:
    model_version = 't5-11b'
else:
    model_version = 't5-small'

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)

data_name = args.data_dir.split('/')[-2]

def load_user2rank_list(file_path):
    """
    Load user2rank_list from a text file.

    Args:
        file_path (str): Path to the file containing user2rank_list data.

    Returns:
        dict: A dictionary where keys are users and values are rank lists.
    """
    user2rank_list = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)  # Split user and rank_list
            user = int(parts[0])  # The user ID
            rank_list = eval(parts[1]) if len(parts) > 1 else []  # Convert rank_list back to Python list
            user2rank_list[user] = rank_list
            if len(user2rank_list) >= args.num_test_samples:
                break
    return user2rank_list

tokenizer = T5Tokenizer.from_pretrained(model_version)
seq_corpus = SeqDataLoader(args.data_dir)
nitem = len(seq_corpus.id2item)
seq_iterator = SeqBatchify(seq_corpus.user2items_positive, tokenizer, args.batch_size)
user2item_test = {}
interacted_items = {}
for user, item_list in seq_corpus.user2items_positive.items():
    user2item_test[user] = [int(item_list[-1])]
    interacted_items[user] = [int(item_id) for item_id in item_list[:-1]]
    if len(user2item_test) >= args.num_test_samples:
        break

# Example usage
file_path = f'{data_name}_user2rank_list.txt'
user2rank_list = load_user2rank_list(file_path)


top_ns = [5, 10]
if args.top_n >= 5:
    for i in range(1, (args.top_n // 5) + 1):
        top_ns.append(i * 5)

for top_n in top_ns:
    hr = evaluate_hr(user2item_test, user2rank_list, top_n)
    print(now_time() + 'HR@{} {:7.4f}'.format(top_n, hr))
    ndcg = evaluate_ndcg(user2item_test, user2rank_list, top_n)
    print(now_time() + 'NDCG@{} {:7.4f}'.format(top_n, ndcg))
    # write txt