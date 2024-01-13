import torch
from graph_transformer_pytorch import GraphTransformer
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_atoms, num_node_features, device=torch.device("cpu")):
        super().__init__()
        self.gt = GraphTransformer(
            dim=num_node_features + 1,
            depth=6,
            edge_dim=3,
            with_feedforwards=True,
            gated_residual=True,
        )
        self.linear = nn.Linear(num_node_features + 1, 1)
        self.atom_embedding = nn.Embedding(num_atoms, num_node_features)
        self.device = device

    def nodes_embedding(self, indices, noise_levels):
        noise_levels = (
            noise_levels.unsqueeze(1)
            .unsqueeze(2)
            .expand(indices.shape[0], indices.shape[1], -1)
        )
        encodings = self.atom_embedding(indices)
        embedding = torch.cat((encodings, noise_levels), dim=2)
        return embedding

    def edges_embedding(self, bonds, coordinates):
        num_atoms = coordinates.shape[1]
        edges = torch.zeros(
            (len(coordinates), num_atoms, num_atoms, 3), dtype=torch.float32
        )
        for n in range(len(edges)):
            for bond in bonds:
                i, j = int(bond[0]), int(bond[1])
                edges[n, i, j] = coordinates[n, i] - coordinates[n, j]
                edges[n, j, i] = coordinates[n, j] - coordinates[n, i]
        return edges.to(self.device)

    def forward(self, indices, coordinates, bonds, noise_levels):
        indices = indices.to(self.device)
        coordinates = coordinates.to(self.device)
        coordinates.requires_grad_()
        bonds = bonds.to(self.device)
        nodes = self.nodes_embedding(indices, noise_levels)
        edges = self.edges_embedding(bonds, coordinates)
        mask = torch.ones((nodes.shape[0], nodes.shape[1])).bool().to(self.device)
        nodes, edges = self.gt(nodes, edges, mask=mask)
        nodes, _ = self.gt(nodes, edges, mask=mask)
        energy = self.linear(nodes).sum()
        gradient = torch.autograd.grad(energy, coordinates, create_graph=True)[0]
        return gradient
