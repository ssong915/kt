import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import dgl
import preprocess


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H) # ( nv * ne )
    n_edge = H.shape[1] # ( ne )
    # the weight of the hyperedge
    W = np.ones(n_edge) # ( ne, ne )
    # the degree of the node
    DV = np.sum(H * W, axis=1) # ( nv, )
    # the degree of the hyperedge
    DE = np.sum(H, axis=0) # ( nv * ne )

    invDE = np.mat(np.diag(np.power(DE, -1))) # ( ne * ne )
    DV2 = np.mat(np.diag(np.power(DV, -0.5))) # ( nv * nv )
    W = np.mat(np.diag(W)) # ( ne * ne )
    H = np.mat(H) # ( nv * ne )
    HT = H.T # ( ne * nv )
    
    return H * HT

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        min_float = np.finfo(float).tiny
        G[np.isnan(G)] = min_float
        return G
    
    
def gen_DGLGraph(snapshot_data):
    """gen_data
        snapshot_data(edge별 node set) 기반 
        DGLGraph generate
    """
    
    he = []
    hv = []
    for i, edge in enumerate(snapshot_data):
        for v in edge:
            he.append(i)
            hv.append(v)
    data_dict = {
        ('node', 'in', 'edge'): (hv, he),        
        ('edge', 'con', 'node'): (he, hv)
    }
    
    g = dgl.heterograph(data_dict)
        
    return g

    
def gen_init_data(args, num_node, n_feat, e_feat):
    """
        snapshot 별로 변함 없는 data들을 미리 선언합니다.
        - node 수
        - node feature
        - node feature dim
    """
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'                 
    
    # Snapshot terms
    if args.time_split_type == 'sec':
        if args.dataset_name == 'email-Enron':
                args.freq_size = 1200
        if args.dataset_name == 'email-Eu':
                args.freq_size = 1000
        if args.dataset_name == 'contact-high-school':
                args.freq_size = 30
        if args.dataset_name == 'contact-primary-school':
                args.freq_size = 700
        if args.dataset_name == 'tags-math-sx':
                args.freq_size = 200
        if args.dataset_name == 'tags-ask-ubuntu':
                args.freq_size = 400  
                
    args.nv = num_node # node 수 (node의 최대 index)
    args.input_dim = args.dim_vertex # node feature dim
    
    args.v_feat = torch.rand(args.nv, args.input_dim).to(device)    
    
    return args


def gen_data(args, snapshot_data):
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    data_dict = {}
    
    # Hyperedge terms
    nv = args.nv
    ne = len(snapshot_data)  # snapshot내 edge 수
    args.ne = ne
    args.e_feat = torch.rand(ne, args.dim_edge).to(device)

    # HGNN terms  
    incidence = torch.zeros(ne, nv)
    for edge_idx, node_set in enumerate(snapshot_data):
        for node_idx in node_set:
            incidence[edge_idx, node_idx] += 1
            
    args.h_incidence  = incidence.T
    args.HGNN_G  = generate_G_from_H(args.h_incidence )
    
    # HNHN terms
    args.v_weight  = torch.zeros(nv, 1)
    args.e_weight  = torch.zeros(ne, 1)
    node2sum = defaultdict(list)
    edge2sum = defaultdict(list)
    e_reg_weight = torch.zeros(ne)
    v_reg_weight = torch.zeros(nv)

    for edge_idx, node_set in enumerate(snapshot_data):
        for node_idx in node_set:
            e_wt = args.e_weight [edge_idx]
            e_reg_wt = e_wt ** args.alpha_e
            e_reg_weight[edge_idx] = e_reg_wt
            node2sum[node_idx].append(e_reg_wt)

            v_wt = args.v_weight [node_idx]
            v_reg_wt = v_wt ** args.alpha_v
            v_reg_weight[node_idx] = v_reg_wt
            edge2sum[edge_idx].append(v_reg_wt)

    v_reg_sum = torch.zeros(nv)
    e_reg_sum = torch.zeros(ne)

    for node_idx, wt_l in node2sum.items():
        v_reg_sum[node_idx] = sum(wt_l)
    for edge_idx, wt_l in edge2sum.items():
        e_reg_sum[edge_idx] = sum(wt_l)

    e_reg_sum[e_reg_sum == 0] = 1
    v_reg_sum[v_reg_sum == 0] = 1
    args.e_reg_weight  = torch.Tensor(e_reg_weight).unsqueeze(-1).to(device)
    args.v_reg_sum  = torch.Tensor(v_reg_sum).unsqueeze(-1).to(device)
    args.v_reg_weight  = torch.Tensor(v_reg_weight).unsqueeze(-1).to(device)
    args.e_reg_sum  = torch.Tensor(e_reg_sum).unsqueeze(-1).to(device)
    
    return data_dict

def load_snapshot(args, DATA):
     
    time_window_factor, time_start_factor = 0.10, 0.4
    
    device = 'cuda'
    
    # # 1. get feature and edge index        
    # rf_data = torch.load('./data/{}/feature_hyper-{}.pt'.format(DATA, DATA))
    # r_data = pd.read_csv('./data/{}/hyper_{}.csv'.format(DATA, DATA))
    
    # # 2. make snapshot  
    # data_info = preprocess.get_datainfo(r_data)
    # snapshot_data, max_node_idx = preprocess.get_snapshot(args, data_info, time_window_factor, time_start_factor)
    
    # n_feat = rf_data['node_feature'].to(device)
    # e_feat = rf_data['edge_feature'].to(device)    
    
    # 1. get feature and edge index        
    r_data = pd.read_csv('/home/dake/workspace/HNHN,HGNN/data/{}/new_hyper_{}.csv'.format(DATA, DATA))
    
    # 2. make snapshot  
    data_info = preprocess.get_datainfo(r_data)
    snapshot_data, snapshot_time, max_node_idx = preprocess.get_snapshot(args, data_info, time_window_factor, time_start_factor)
    
    n_feat = 0
    e_feat = 0
    
    return snapshot_data, snapshot_time, max_node_idx, n_feat, e_feat