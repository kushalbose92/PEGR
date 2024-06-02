import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
import os 
import numpy as np
import dgl
import networkx as nx
import sys

from torch_geometric.nn import GCNConv, global_mean_pool
from ginconv import GINConv
from torch_geometric.utils import degree, add_self_loops, remove_self_loops, from_scipy_sparse_matrix, to_undirected, dense_to_sparse, to_dense_adj, to_dense_batch
# from torch.nn.parameter import Parameter
import torch_geometric.transforms as T 
from torch_geometric.data import Data

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import scipy.sparse as sp

# torch.autograd.set_detect_anomaly(True)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(MLP, self).__init__()
        
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, hidden_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    

# step activation function 
class custom_step_function(nn.Module):
    def __init__(self):
        super(custom_step_function, self).__init__()
    
    def forward(self, x):
        x[x>=0] = 1.0
        x[x<0] = 0.0
        return x
    

# defining deep gnn models
class GNN(nn.Module):

    def __init__(self, dataset, model, num_layers, mlp_layers, input_dim, hidden_dim, dropout, th, rewiring, alpha, device):
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.dataset = dataset
        self.model = model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.gnn_convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.th = th
        self.mlp_layers = mlp_layers
        self.rewiring = rewiring
        self.alpha = alpha
        print("alpha ", self.alpha)
        
        # classifier layer
        self.cls = nn.Linear(self.hidden_dim, self.dataset.num_classes)

        # adding parallel edges rewiring
        if int(self.alpha) > 0:
            self.rewiring_model = PEGR(device, self.alpha)
        
        if model == 'gcn':
            print("Using GCN model...")
            for i in range(self.num_layers):
                if i == 0:
                    self.gnn_convs.append(GCNConv(self.dataset.num_features, self.hidden_dim).to(self.device))
                    self.lns.append(nn.LayerNorm(hidden_dim))
                elif i == self.num_layers - 1:
                    self.gnn_convs.append(GCNConv(self.hidden_dim, self.hidden_dim).to(self.device))
                else:
                    self.gnn_convs.append(GCNConv(self.hidden_dim, self.hidden_dim).to(self.device))
                    self.lns.append(nn.LayerNorm(hidden_dim))
        elif model == 'gin':
            print("Using GIN model...")
            # for i in range(num_layers):
            #     if i == 0:
            #         self.gnn_convs.append(GINConv(nn.Sequential(nn.Linear(self.dataset.num_features, self.hidden_dim),nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),nn.Linear(self.hidden_dim, self.hidden_dim))).to(self.device))
            #     else:
            #         self.gnn_convs.append(GINConv(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),nn.Linear(self.hidden_dim, self.hidden_dim))).to(self.device))
            for i in range(num_layers):
                if i == 0:
                    self.gnn_convs.append(GINConv(self.dataset.num_features, self.hidden_dim).to(self.device))
                else:
                    self.gnn_convs.append(GINConv(self.hidden_dim, self.hidden_dim).to(self.device))
        else:
            print("Invalid model name...")

        
    def forward(self, x, edge_index, batch):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)
        
        # x = F.dropout(x, p=self.dropout, training=True)

        dir_energy = 0.0
        edge_ratio = 0.0
        prob = torch.tensor([0.0])
            
        # adding parallel edges 
        if int(self.alpha) > 0:
            rewired_edge_index, updated_edge_weights = self.rewiring_model(x, edge_index, batch)
        else:
            rewired_edge_index = edge_index
            updated_edge_weights = None
       
        if self.model == 'gcn':
            # message propagation through hidden layers
            for i in range(self.num_layers):
                x = self.gnn_convs[i](x, rewired_edge_index, updated_edge_weights)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = self.lns[i](x)
                    x = F.dropout(x, p=self.dropout, training=True)
        else:
            adj_matrix = torch.zeros(x.shape[0], x.shape[0]).to(self.device)
            if int(self.alpha) > 0:
                adj_matrix[rewired_edge_index[0], rewired_edge_index[1]] = updated_edge_weights
            else:
                adj_matrix[rewired_edge_index[0], rewired_edge_index[1]] = torch.ones(rewired_edge_index.shape[1]).to(self.device)
            for i in range(self.num_layers):
                # x = self.gnn_convs[i](x, rewired_edge_index)
                x = self.gnn_convs[i](x, adj_matrix)
                
        # applying mean pooling
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=True)
        x = self.cls(x)

        embedding = x
        x = F.log_softmax(x, dim = 1)
        return embedding, x
        


# Increase of spectral gap by adding parallel edges
class PEGR(nn.Module):
    def __init__(self, device, alpha):
        super(PEGR, self).__init__()
        self.device = device
        self.alpha = alpha
    
    def forward(self, x, edge_index, batch):
        num_nodes = x.shape[0]
        # print("features ", x.shape)
        num_graphs = max(batch).item() + 1
        # _, mask = to_dense_batch(x, batch) 

        # edge_batch = batch[edge_index[0].detach().cpu().numpy()]
        # print("num edges ", edge_batch.shape)
        # for e in range(edge_index.shape[1]):
        #     print(edge_index[0][e], "   ", edge_index[1][e], "   ", edge_batch[e])
        
        
        # adj_matrix = to_dense_adj(edge_index, batch = batch, max_num_nodes = num_nodes)
        # adj_matrix = torch.tensor(adj_matrix, dtype=torch.long)
        

        # updated_batch_adj = torch.zeros(num_graphs, num_nodes, num_nodes).to(self.device)
        # rewired_edge_index = None
        # updated_edge_weights = None
        
                
        # for g in range(num_graphs):
        #     A_hat = adj_matrix[g]
        #     # print("g ", g, " ", A_hat.sum(), "  ", A_hat.shape[0])
        #     A_hat = int(self.alpha) * A_hat
        #     graph_wise_mask = torch.tensor([1 if batch[i] == g else 0 for i in range(num_nodes)], dtype=torch.float).unsqueeze(1).to(self.device)
        #     num_nodes_g = torch.sum(torch.eq(batch, g))
        #     # print(graph_wise_mask.shape)
        #     graph_adj_mask = torch.matmul(graph_wise_mask, torch.t(graph_wise_mask))
        #     # print("masks sum ", graph_adj_mask.shape, " ", graph_adj_mask.sum(), "  ", num_nodes_g)
        #     # for row in graph_adj_mask:
        #     #     print(row)
        #     # graph_adj_mask = torch.triu(graph_adj_mask, diagonal=1)
        #     A_hat = A_hat * graph_adj_mask
            
        #     # print(graph_adj_mask)
            
        #     # adding self-loops
        #     # A_hat = 2 * A_hat
        #     # updated_batch_adj[g] = A_hat
        #     # for row in A_hat:
        #     #     print(row)
        #     # print(A_hat.shape)
        #     edge_indices = torch.where(A_hat != 0)
        #     flatten_adj = A_hat.reshape(num_nodes * num_nodes)
        #     edge_weights = flatten_adj[flatten_adj != 0]
        #     # print("index ", g, "  ", edge_weights, "  ", edge_indices)
        #     updated_edge_index = torch.stack([edge_indices[0], edge_indices[1]])
        #     # updated_edge_index = updated_edge_index.repeat(1, 5)
           
        #     if rewired_edge_index is None:
        #         rewired_edge_index = updated_edge_index
        #         updated_edge_weights = edge_weights
        #     else:
        #         rewired_edge_index = torch.cat([rewired_edge_index, updated_edge_index], dim = 1)
        #         updated_edge_weights = torch.cat([updated_edge_weights, edge_weights], dim = 0)
                
        num_edges = edge_index.shape[1]
        updated_edge_weights = torch.ones(num_edges).to(self.device) * int(self.alpha+1)
        return edge_index, updated_edge_weights
    
        
