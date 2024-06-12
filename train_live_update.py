import random
from collections import defaultdict
import dgl
import torch 
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import AveragePrecision
from sklearn import metrics
import os
from tqdm import tqdm
import pandas as pd


import utils, preprocess, data_load, models, decoder

def model_train(args, train_snapshot_edges, next_snapshot_pos_edges, next_snapshot_neg_edges, st_model, decoder, optimizer, scheduler):  
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'

    Aggregator = args.aggregator
    
    st_model.train()
    decoder.train()
    optimizer.zero_grad()
    
    all_v_feat = args.v_feat
    
    data_dict = data_load.gen_data(args, train_snapshot_edges)
    reindexed_snapshot_edges, snapshot_node_index = utils.reindex_snapshot(train_snapshot_edges)
    g = data_load.gen_DGLGraph(reindexed_snapshot_edges).to(device)   
    train_pos_edges, train_neg_edges = utils.get_samples(next_snapshot_pos_edges, next_snapshot_neg_edges, len(train_snapshot_edges))
    
    # 1. Update hypergraph node features (message passing)
    if args.model == 'gcn':
        pass
    elif args.model == 'hgnn':
        v_feat = args.v_feat
        incidence = data_dict['HGNN_G']
        matrix_np = np.array(incidence)
        incidence = matrix_np.tolist()
        incidence = torch.tensor(incidence)
        incidence = incidence.to(device) 
        for l in range(args.num_layers):
            v_feat = st_model(v_feat, incidence)        
        all_v_feat = v_feat
    elif args.model == 'hnhn':
        v_feat = all_v_feat[snapshot_node_index].to(device)
        e_feat = data_dict['e_feat'][g.nodes('edge')].to(device)
        v_reg_weight = data_dict['v_reg_weight'][snapshot_node_index].to(device)
        v_reg_sum = data_dict['v_reg_sum'][snapshot_node_index].to(device)
        e_reg_weight = data_dict['e_reg_weight'][g.nodes('edge')].to(device)
        e_reg_sum = data_dict['e_reg_sum'][g.nodes('edge')].to(device)
        
        for l in range(args.num_layers):
            v_feat = st_model(g, v_feat, e_feat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum)
            
        all_v_feat[snapshot_node_index] = v_feat # ~Ht_l            
        g = g.to('cpu')
        
    args.v_feat = all_v_feat
    
    # 2. Hyperedge prediction  
    # t+1 시점의 positive hyperedge의 정보를 기반으로 negative hyperedge 생성        
    pos_preds = decoder(all_v_feat.detach(), train_pos_edges, Aggregator)
    pos_preds = pos_preds.squeeze()
    
    neg_preds = decoder(all_v_feat.detach(), train_neg_edges, Aggregator)
    neg_preds = neg_preds.squeeze().to(device)
    
    pos_labels = torch.ones_like(pos_preds)
    neg_labels = torch.zeros_like(neg_preds)

    # 3. Compute training loss and update parameters
    criterion = nn.MSELoss()
    real_loss = criterion(pos_preds, pos_labels)
    fake_loss = criterion(neg_preds, neg_labels)
    train_loss = real_loss + fake_loss
    if args.neg_mode == 'none':
        train_loss = real_loss
    
    
    train_loss.backward()   
    optimizer.step()
    
    return train_loss.item()


def model_eval(args, dataloader, next_snapshot_pos_edges, next_snapshot_neg_edges, st_model, decoder):
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    Aggregator = args.aggregator
    
    st_model.eval()
    decoder.eval()
    
    all_v_feat = args.v_feat
    
    preds = []
    labels = []
    
    while True:
        val_snapshot_edges, time_end = dataloader.next()
        valid_pos_edges, valid_neg_edges = utils.get_samples(next_snapshot_pos_edges, next_snapshot_neg_edges, len(val_snapshot_edges))
                        
        data_dict = data_load.gen_data(args, val_snapshot_edges)
        reindexed_snapshot_edges, snapshot_node_index = utils.reindex_snapshot(val_snapshot_edges)
        g = data_load.gen_DGLGraph(reindexed_snapshot_edges).to(device) 
        
        if args.model == 'gcn':
            pass
        elif args.model == 'hgnn':
            v_feat = args.v_feat
            incidence = data_dict['HGNN_G']
            matrix_np = np.array(incidence)
            incidence = matrix_np.tolist()
            incidence = torch.tensor(incidence)
            incidence = incidence.to(device) 
            for l in range(args.num_layers):
                v_feat = st_model(v_feat, incidence)   
            all_v_feat = v_feat
        elif args.model == 'hnhn':
            v_feat = all_v_feat[snapshot_node_index].to(device)
            e_feat = data_dict['e_feat'][g.nodes('edge')].to(device)
            v_reg_weight = data_dict['v_reg_weight'][snapshot_node_index].to(device)
            v_reg_sum = data_dict['v_reg_sum'][snapshot_node_index].to(device)
            e_reg_weight = data_dict['e_reg_weight'][g.nodes('edge')].to(device)
            e_reg_sum = data_dict['e_reg_sum'][g.nodes('edge')].to(device)
            for l in range(args.num_layers):
                v_feat = st_model(g, v_feat, e_feat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum)
                
            all_v_feat[snapshot_node_index] = v_feat
            g = g.to('cpu')
                    
        pos_preds = decoder(all_v_feat, valid_pos_edges, Aggregator)
        neg_preds = decoder(all_v_feat, valid_neg_edges, Aggregator)
        
        pos_labels = torch.ones_like(pos_preds)
        neg_labels = torch.zeros_like(neg_preds)
        
        # 3. Compute training loss and update parameters
        preds += ( pos_preds.tolist() + neg_preds.tolist() )
        labels += ( pos_labels.tolist() + neg_labels.tolist() )

        if time_end:
            break     

    return preds, labels
    

def train(args, st_model, snapshot_data, decoder, optimizer, scheduler, f_log, j):
    total_time = len(snapshot_data)  
        
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    patience = args.early_stop
    
    auc_roc_list = []
    ap_list = []
        
    for t in tqdm(range(total_time-1)):  # batch.next      
        best_roc = 0         
        best_vfeat = args.v_feat

        if t != 0:
            # load t-1 best model parameter
            st_model.load_state_dict(torch.load(f"/home/dake/workspace/HNHN,HGNN/data/{args.dataset_name}/{args.model}_model_{j}.pkt"))
            decoder.load_state_dict(torch.load(f"/home/dake/workspace/HNHN,HGNN/data/{args.dataset_name}/{args.model}_decoder_{j}.pkt"))
            
        # load t snapshot (for training)    
        train_dataloader, valid_dataloader, test_dataloader = preprocess.get_dataloaders(snapshot_data[t], args.batch_size, device) 
        
        # load t+1 snapshot (for link prediction)
        next_snapshot_pos_edges, next_snapshot_neg_edges = preprocess.load_hedge_pos_neg(snapshot_data[t+1], args.neg_mode)      
                    
        for epoch in (range(args.epochs)):     
            # [Train]      
            while True:            
                train_snapshot_edges, time_end = train_dataloader.next()          
                loss = model_train(args, train_snapshot_edges, next_snapshot_pos_edges, next_snapshot_neg_edges,
                                                        st_model, decoder, optimizer, scheduler)                  
                if time_end:
                    break       

            # [Validation for early stop]                           
            val_preds, total_labels = model_eval(args, valid_dataloader, next_snapshot_pos_edges, next_snapshot_neg_edges, st_model, decoder)
            auc_roc, ap = utils.measure(total_labels, val_preds)
            
            f_log.write(f"{t} time - {epoch} epoch, Training loss : {loss} / NS : Val AP : {ap} / AUROC : {auc_roc}\n")   
            print(f"{t} time - {epoch} epoch, Training loss : {loss} / NS : Val AP : {ap} / AUROC : {auc_roc}\n")     
                    
            if best_roc < auc_roc or epoch == 0:
                best_roc = auc_roc
                best_ap = ap  
                best_vfeat = args.v_feat         
                torch.save(st_model.state_dict(), f"/home/dake/workspace/HNHN,HGNN/data/{args.dataset_name}/{args.model}_model_{j}.pkt")
                torch.save(decoder.state_dict(), f"/home/dake/workspace/HNHN,HGNN/data/{args.dataset_name}/{args.model}_decoder_{j}.pkt")
                no_improvement_count = 0    
                if epoch == args.epochs:
                    args.v_feat = best_vfeat
                    break      
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    args.v_feat = best_vfeat
                    
                    # print()
                    # print(f"No improvement for {patience} epochs. Early stopping...")
                    break 
        
        st_model.load_state_dict(torch.load(f"/home/dake/workspace/HNHN,HGNN/data/{args.dataset_name}/{args.model}_model_{j}.pkt"))
        decoder.load_state_dict(torch.load(f"/home/dake/workspace/HNHN,HGNN/data/{args.dataset_name}/{args.model}_decoder_{j}.pkt"))
            
        # [Test]            
        test_preds, test_labels = model_eval(args, test_dataloader, next_snapshot_pos_edges, next_snapshot_neg_edges, st_model, decoder)
        test_auc_roc, test_ap = utils.measure(test_labels, test_preds)  
        
        auc_roc_list.append(test_auc_roc)
        ap_list.append(test_ap)         
        
        f_log.write(f"[{t} time Val AP : {best_roc} / AUROC : {best_ap} ]\n")
        f_log.write(f"[{t} time Test AP : {test_auc_roc} / AUROC : {test_ap} ]\n")
        
        print(f"[{t} time Val AP : {best_roc} / AUROC : {best_ap} ]\n")
        print(f"[{t} time Test AP : {test_auc_roc} / AUROC : {test_ap} ]\n")
        
    final_auc_roc = sum(auc_roc_list)/len(auc_roc_list)
    final_ap = sum(ap_list)/len(ap_list)
    return final_auc_roc, final_ap 