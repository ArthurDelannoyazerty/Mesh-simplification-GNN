import torch
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.data import Data

class DevConvGNN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__(aggr='max')
        self.linear_theta = Linear(in_channels, hidden_channels)
        self.linear_phi = Linear(hidden_channels, out_channels)

    def forward(self, edge_index, features):
        maxi = self.propagate(edge_index, x=features)
        w_phi_output = self.linear_phi(maxi)
        return w_phi_output

    def message(self, x_i, x_j):
        w_theta_output: torch.Tensor = self.linear_theta(x_i - x_j)
        return w_theta_output
