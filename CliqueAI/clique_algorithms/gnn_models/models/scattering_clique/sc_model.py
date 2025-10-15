import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.utils.convert import to_scipy_sparse_matrix

from .packages.gnn import GNN
from .packages.utils import compute_node_features
from .packages.sampler import getclique


class ScatteringCliqueModel:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = self.__load_model()

    def __load_model(self):
        model = GNN(
            input_dim=3,
            hidden_dim=8,
            output_dim=1,
            n_layers=4,
        )
        model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        model.eval()
        return model

    def prepare_data(self, num_nodes: int, adjacency_list: list[list]) -> Data:
        dict_of_lists = {i: adjacency_list[i] for i in range(num_nodes)}
        edge_list = [
            [i, j] for i, neighbors in dict_of_lists.items() for j in neighbors if j > i
        ]
        edge_index = to_undirected(torch.tensor(edge_list, dtype=torch.long).T)
        feature_vector = compute_node_features(edge_list, num_nodes)
        x = torch.tensor(feature_vector, dtype=torch.float)
        return Data(x=x, edge_index=edge_index)

    def edge_index_to_sparse(self, edge_index, num_nodes, device):
        """
        Convert edge_index to sparse adjacency tensor on given device.
        """
        values = torch.ones(edge_index.size(1), device=device)
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
        return adj.coalesce()

    def predict_iter(self, num_nodes: int, adjacency_list: list[list[int]]) -> list[int]:
        data: Data = self.prepare_data(num_nodes, adjacency_list)
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)

        adj = self.edge_index_to_sparse(edge_index, num_nodes, device=self.device)
        adjmatrix = to_scipy_sparse_matrix(edge_index)

        output = self.model(x, adj, moment=1, device=self.device)

        num_walkers = min(10, num_nodes)
        sample_length = min(90, num_nodes)

        for w in range(num_walkers):
            yield getclique(adjmatrix, output.cpu(), w, sample_length)

    def predict(self, num_nodes: int, adjacency_list: list[list[int]]) -> list[list[int]]:
        return [list(map(int, c)) for c in self.predict_iter(num_nodes, adjacency_list)]