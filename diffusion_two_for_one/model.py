import torch
from graph_transformer_pytorch import GraphTransformer
import torch.nn as nn

class Net(nn.Module):

    def __init__(
            self,
            num_atoms,
            num_node_features
            ):
        super().__init__()
        self.gt = GraphTransformer(
            dim = num_node_features+1,
            depth = 6,
            edge_dim = 3,
            with_feedforwards = True,
            gated_residual = True,
        )
        self.linear = nn.Linear(num_node_features+1, 1)
        self.atom_embedding = nn.Embedding(num_atoms, num_node_features)

    def nodes_embedding(
            self,
            coordinates,
            noise_level
            ):
        indices = [i for i in range(len(coordinates))]
        encodings = self.atom_embedding(torch.tensor(indices, dtype=torch.long))
        noise_levels = torch.full((1, len(coordinates), 1), fill_value=noise_level)
        embedding = torch.cat((encodings, noise_levels.squeeze(0)), dim=1)
        return embedding.unsqueeze(0)

    def edges_embedding(
            self,
            bonds,
            coordinates
            ):
        num_atoms = len(coordinates)
        edges = torch.zeros((num_atoms, num_atoms, 3), dtype=torch.float32)
        for bond in bonds:
            i, j = bond[0], bond[1]
            edges[i, j] = coordinates[i] - coordinates[j]
            edges[j, i] = coordinates[j] - coordinates[i]
        return edges.unsqueeze(0)

    def forward(
            self,
            coordinates,
            bonds,
            noise_level
            ):
        nodes = self.nodes_embedding(coordinates, noise_level)
        edges = self.edges_embedding(bonds, coordinates)
        mask = torch.ones(1, len(coordinates)).bool()
        nodes, _ = self.gt(nodes, edges, mask=mask)
        energy = self.linear(nodes)
        return energy
    