import math
import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
# from torch.nn.modules.module import Module


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


# define a single graph convolution layer
class GINConv(torch.nn.Module):

    def __init__(self, in_features, out_features, epsilon=0.0):
        super(GINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = epsilon
        
        self.mlp = MLP(self.in_features, self.out_features, 1, 0.50)

    def forward(self, input, adj):
        I = torch.eye(adj.shape[0]).to(adj.device)
        adj = adj + (1 + self.epsilon) * I
        support = torch.matmul(adj, input) 
        output = self.mlp(support)
        return output