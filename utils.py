
import numpy as np
import torch
import random
import argparse
import warnings
import logging

import math, sys
import numpy as np

from sklearn import metrics
from torchmetrics import AveragePrecision
from torchmetrics.retrieval import RetrievalMRR

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Hypergraph representation learning')

    ##### training hyperparameter #####
    parser.add_argument("--dataset_name", type=str, default='email-Eu', help='dataset name: _')
    parser.add_argument('--seed', type=int, default=1111, metavar='S', help='Random seed (default: 1111)')
    parser.add_argument("--folder_name", default='exp1', help='experiment number')
    parser.add_argument("--gpu", type=int, default=1, help='gpu number. -1 if cpu else gpu number')
    parser.add_argument("--exp_num", default=1, type=int, help='number of experiments')
    parser.add_argument("--epochs", default=50, type=int, help='number of epochs')
    parser.add_argument("--early_stop", default=10, type=int, help='number of early stop')
    parser.add_argument("--training", type=str, default='wgan', help='loss objective: wgan, none')
    parser.add_argument("--lr", default=0.0001, type=float, help='learning rate')
    parser.add_argument("--neg_size", default=100, type=int, help='learning rate')
    
    parser.add_argument("--eval_mode", default='train_fixed_split', type=str, help='learning rate')
    
    # snapshot split
    parser.add_argument("--time_split_type", default='sec', type=str, help='snapshot split type')
    parser.add_argument("--batch_size", default=32, type=int, help='batch size')
    parser.add_argument("--freq_size", default=2628000, type=int, help='batch size')
    
    # encoder
    parser.add_argument("--model", default='hnhn', type=str, help='encoder: hgnn')
    parser.add_argument("--num_layers", default=2, type=int, help='number of layers')
    parser.add_argument("--alpha_e", default=0, type=float, help='normalization term for hnhn')
    parser.add_argument("--alpha_v", default=0, type=float, help='normalization term for hnhn')
        
    # decoder
    parser.add_argument("--neg_mode", type=str, default='sns', help='negative sampling: mns, sns, cns')
    parser.add_argument("--aggregator", type=str, default='Maxmin', help='aggregator: maxmin, average, attention')
    
    # feature dimmension
    parser.add_argument("--dim_hidden", default=128, type=int, help='dimension of hidden vector')
    parser.add_argument("--dim_vertex", default=128, type=int, help='dimension of vertex hidden vector')
    parser.add_argument("--dim_edge", default=128, type=int, help='dimension of edge hidden vector')
    parser.add_argument("--dim_time", default=128, type=int, help='dimension of time hidden vector')
    
    args = parser.parse_args()
    
    return args

def get_num_iters(data_dict, batch_size: int, label: str = 'Train'):
    if label == 'Train':
        train_iters = math.ceil(len(data_dict["train_only_pos"] + data_dict["ground_train"])/batch_size)
        val_pos_iters = math.ceil(len(data_dict["valid_only_pos"] + data_dict["ground_valid"])/batch_size)
        val_neg_iters = math.ceil(len(data_dict["valid_sns"])/batch_size)

        return train_iters, val_pos_iters, val_neg_iters

    elif label == 'Test':
        test_pos_iters = math.ceil(len(data_dict["test_pos"])/batch_size)
        test_neg_iters = math.ceil(len(data_dict["test_sns"])/batch_size)

        return test_pos_iters, test_neg_iters

    else:
        sys.exit( "Wrong Label Name! 'Train or Test'")

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
@torch.no_grad()    
def measure(label, pred):
    average_precision = AveragePrecision(task='binary')
    auc_roc = metrics.roc_auc_score(np.array(label), np.array(pred))

    label = torch.tensor(label)
    label = label.type(torch.int64)
    ap = average_precision(torch.tensor(pred), torch.tensor(label))
                    
    return auc_roc, ap

def reindex_snapshot(snapshot_edges):
    org_node_index = []
    reindex_snapshot_edges = [[0 for _ in row] for row in snapshot_edges]
    for i, edge in enumerate(snapshot_edges):
        for j, node in enumerate(edge):
            if node not in org_node_index:
                org_node_index.append(node)
            new_idx = org_node_index.index(node)
            reindex_snapshot_edges[i][j] = new_idx
    
    return reindex_snapshot_edges, org_node_index

def split_edges(snapshot_edges):
    
    time_hyperedge = list(snapshot_edges)
    total_size = len(time_hyperedge)
    idcs = np.arange(len(time_hyperedge)).tolist()
    
    test_label_size = int(math.ceil(total_size * 0.1))
    test_size = int(math.ceil(total_size * 0.1))
    valid_size = int(math.ceil(total_size * 0.2))
    
    test_label = time_hyperedge[-test_label_size:]
    test_label = [item for sublist in test_label for item in sublist]
    
    test_hyperedge = time_hyperedge[-(test_label_size + test_size):-test_label_size]
    test_hyperedge = [item for sublist in test_hyperedge for item in sublist]
    
    valid_hyperedge = time_hyperedge[-(test_label_size + test_size + valid_size):-(test_label_size + test_size)]
    valid_hyperedge = [item for sublist in valid_hyperedge for item in sublist]
    
    train_hyperedge = time_hyperedge[:-(test_label_size + test_size + valid_size)]
    
    
    return train_hyperedge, valid_hyperedge, test_hyperedge, test_label


def get_samples(pos_hyperedges, neg_hyperedges, batch_size):
    if len(pos_hyperedges) >= batch_size:
        # 다음 snapshot의 크기가 batch size보다 클 경우 (다음 snapshot에서 뽑을 수 있는 경우)
        # 중복 없이 
        return random.sample(pos_hyperedges, batch_size), random.sample(neg_hyperedges, batch_size)
    else:
        # 다음 snapshot의 크기가 batch size보다 작을 경우
        # 중복 포함
        return random.choices(pos_hyperedges, k=batch_size), random.choices(neg_hyperedges, k=batch_size)