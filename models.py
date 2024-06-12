import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import math

from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from dgl import DGLGraph


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, n_feat : torch.Tensor, incidence: torch.Tensor):
        n_feat = n_feat.matmul(self.weight) 
        if self.bias is not None:
            n_feat = n_feat + self.bias
        n_feat = incidence.matmul(n_feat)
        return n_feat

class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5): #in_ch: node feature, n_hid: dim_hidden
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgnn_layer_norm = nn.LayerNorm(n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, n_feat, incidence):
        # n_feat: node 수 * node feature 수
        # incidence: node 수 * node 수

        # hgc1) node 수 * node feature --(weight: node feature * hidden dim)--> node 수 * hidden dim
        n_feat = F.relu(self.hgc1(n_feat, incidence)) 
        n_feat = F.dropout(n_feat, self.dropout)
        n_feat = self.hgnn_layer_norm(n_feat)
        # hgc2) node 수 * hidden dim --(weight: hidden dim * hidden dim)--> node 수 * hidden dim
        n_feat = self.hgc2(n_feat, incidence) 
        return n_feat

class HNHN(nn.Module):
    def __init__(self, input_dim, dim_vertex, dim_edge):
        super(HNHN, self).__init__()
        self.vtx_lin_1layer = torch.nn.Linear(input_dim, dim_vertex)
        self.vtx_lin = torch.nn.Linear(dim_vertex, dim_vertex)
        
        self.ve_lin = torch.nn.Linear(dim_vertex, dim_edge)
        self.ev_lin = torch.nn.Linear(dim_edge, dim_vertex)
        
        self.edge_norm = nn.LayerNorm(dim_edge)
        self.node_norm = nn.LayerNorm(dim_vertex)
        
    def weight_fn(self, edges):
        weight = edges.src['reg_weight']/edges.dst['reg_sum']
        return {'weight': weight}
    
    def message_func(self, edges):
        return {'Wh': edges.src['Wh'], 'weight': edges.data['weight']}

    def reduce_func(self, nodes):
        Weight = nodes.mailbox['weight']
        aggr = torch.sum(Weight * nodes.mailbox['Wh'], dim=1)
        return {'h': aggr}

    def forward(self, g, vfeat, efeat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum):

        with g.local_scope():
            feat_v = self.vtx_lin_1layer(vfeat)
            feat_e = efeat

            g.ndata['h'] = {'node': feat_v}
            g.ndata['Wh'] = {'node' : self.ve_lin(feat_v)}
            g.ndata['reg_weight'] = {'node':v_reg_weight, 'edge':e_reg_weight}
            g.ndata['reg_sum'] = {'node':v_reg_sum, 'edge':e_reg_sum}
            
            # edge aggregation
            g.apply_edges(self.weight_fn, etype='in')
            g.update_all(self.message_func, self.reduce_func, etype='in')            
            feat_e = g.ndata['h']['edge']
            
            g.ndata['Wh'] = {'edge' : self.ev_lin(feat_e)}
            
            # node aggregattion
            g.apply_edges(self.weight_fn, etype='con')
            g.update_all(self.message_func, self.reduce_func, etype='con')
            feat_v = g.ndata['h']['node']            
            
            return feat_v
