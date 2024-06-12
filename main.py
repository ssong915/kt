from collections import defaultdict
import random, os, sys
import numpy as np
import pandas as pd
import time, statistics
import logging, warnings

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn import metrics
import utils, models, data_load, train_fixed_split, preprocess, train_live_update
from decoder import *
import pdb
from dgl import DGLGraph

warnings.simplefilter("ignore")

def evaluate(model, g, n_feat, he_feat, dataloader, iters, method):
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for _ in range(iters):
            # 1. HNHN message passing
            dummy_mask = utils.gen_feature_mask(0)
            nfeat, efeat = model(g, dummy_mask, n_feat, he_feat)

            # 2. candidate scoring for hyperedge in validation/test datasets
            hedges, labels = dataloader.next() 
            test_preds += model.aggregate(nfeat, hedges, mode='Eval', method=method) 
            test_labels.append(labels.detach())

        test_preds = torch.sigmoid(torch.stack(test_preds).squeeze())
        test_labels = torch.cat(test_labels, dim=0)

    return test_preds.tolist(), test_labels.tolist()
    
def train(args):
    #---------------- get args -------------------#
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    DATA = args.dataset_name
            
                
    os.makedirs(f"/data/checkpoints/{args.folder_name}", exist_ok=True)
    test_env = f'logs/{args.model}/{DATA}_{args.eval_mode}_{args.model}'
            
    os.makedirs(f"logs/{args.model}", exist_ok=True)
    args.folder_name = test_env
    
    if args.time_split_type == 'sec': 
        f_log = open(f"{args.folder_name}.log", "w")
        f_log.write(f"args: {args}\n")
            
    roc_list = []
    ap_list = []   
        
    # Load data
    snapshot_data, snapshot_time, num_node, n_feat, e_feat = data_load.load_snapshot(args, DATA)
    args = data_load.gen_init_data(args, num_node, n_feat, e_feat)
    
    for i in range(args.exp_num): # number of splits (default: 5)
        # Change the random seed 
        i = 111
        utils.set_random_seeds(i)

        print(f'============================================ Experiments {i} ==================================================')
        f_log.write(f'============================================ Experiments {i} ==================================================')
        # Initialize models
        # 1. Hypergraph encoder
        if args.model == 'gcn':
            st_model = models.GCN(input_dim= args.input_dim, hidden_dim=args.dim_vertex, dropout=0.5)
        elif args.model == 'hgnn':
            st_model = models.HGNN(in_ch= args.input_dim, n_hid=args.dim_vertex, dropout=0.5)
        elif args.model == 'hnhn':
            st_model = models.HNHN(args.input_dim, args.dim_vertex, args.dim_edge)
        
        # encoder = HypergraphEncoder
        
        # # 2. Spatio temporalLayer
        # st_updater = models.SpatioTemporalLayer(dim_in= args.input_dim, dim_out=args.dim_vertex)
        # # st_model = models.OurModel2(encoder, args.num_layers).to(device)
        # if args.model == 'hgnn':
        #     st_model = models.OurModel(encoder, st_updater, args.num_layers).to(device)
        # elif args.model == 'hnhn':
            # st_model = models.OurModel_HNHN(encoder, st_updater, args.num_layers).to(device)
        st_model = st_model.to(device)
        #3. Decoder (classifier) for hyperedge prediction
        cls_layers = [args.dim_vertex, 128, 8, 1]
        decoder = Decoder(cls_layers).to(device)
                
        optimizer = torch.optim.RMSprop(list(st_model.parameters()) + list(decoder.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        if args.eval_mode == 'train_fixed_split' :
            auc_roc, ap = train_fixed_split.train(args, st_model, snapshot_data, decoder, optimizer, scheduler, f_log, i)
        else :
            auc_roc, ap = train_live_update.train(args, st_model, snapshot_data, decoder, optimizer, scheduler, f_log, i)
        roc_list.append(auc_roc)
        ap_list.append(ap)
        
        print(' ')  
         
    final_roc = sum(roc_list)/len(roc_list)
    final_ap = sum(ap_list)/len(ap_list)

    if args.exp_num > 1:
        std_roc = statistics.stdev(roc_list)
        std_ap = np.std(ap_list)
    else:
        std_roc = 0.0 
        std_ap = 0.0 

    print('============================================ Test End =================================================')

    print('[ AVG SNAPSHOT ]')
    print('AUROC\t AP\t ')
    print(f'{final_roc:.4f}\t{final_roc:.4f}')
    
    f_log.write(f'{final_roc:.4f}\t{final_ap:.4f}')
    f_log.write(f"\nNS : avg AP : {final_ap} / avg AUROC : {final_roc}\n")
    f_log.write(f'============================================ Test End =================================================')
    
    f_log.close

if __name__ == '__main__':
    args = utils.parse_args()
    train(args)