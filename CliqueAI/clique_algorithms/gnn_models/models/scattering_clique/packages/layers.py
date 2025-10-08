import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from .utils import sparse_mx_to_torch_sparse_tensor


class GC_withres(Module):
    def __init__(self, in_features, out_features, smooth):
        super(GC_withres, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.smooth = smooth
        self.mlp = nn.Linear(in_features, out_features)

    def forward(self, input, adj, device="cuda"):
        support = self.mlp(input)
        I_n = sp.eye(adj.size(0))
        I_n = sparse_mx_to_torch_sparse_tensor(I_n).to(device)
        A_gcn = adj + I_n
        degrees = torch.sparse.sum(A_gcn, 0)
        D = degrees
        D = D.to_dense()
        D = torch.pow(D, -0.5)
        D = D.unsqueeze(dim=1)
        A_gcn_feature = support
        A_gcn_feature = torch.mul(A_gcn_feature, D)
        A_gcn_feature = torch.spmm(A_gcn, A_gcn_feature)
        A_gcn_feature = torch.mul(A_gcn_feature, D)
        output = A_gcn_feature * self.smooth + support
        output = output / (1 + self.smooth)
        return output