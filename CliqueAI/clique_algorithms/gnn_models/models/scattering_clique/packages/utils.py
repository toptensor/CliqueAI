import igraph as ig
import numpy as np
import torch

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def compute_node_features(edge_list, num_nodes):
    """Compute node features: [eccentricity, log(degree), clustering coefficient]"""
    g = ig.Graph(n=num_nodes, edges=edge_list, directed=False)

    ecc = g.eccentricity()
    deg = [np.log(d) if d > 0 else 0.0 for d in g.degree()]
    clust = g.transitivity_local_undirected(vertices=None, mode="zero")

    return np.stack([ecc, deg, clust], axis=1)