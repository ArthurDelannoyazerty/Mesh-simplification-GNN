import torch
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.data import Data

class StackedTriConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.triconv1 = TriConv(in_channels, hidden_channels)
        self.triconv2 = TriConv(hidden_channels, hidden_channels)
        self.triconv3 = TriConv(hidden_channels, out_channels)

    def forward(self, edge_index, r_matrix, f):
        f = self.triconv1(edge_index, r_matrix, f).relu()
        f = self.triconv2(edge_index, r_matrix, f).relu()
        f = self.triconv3(edge_index, r_matrix, f).softmax()
        return f



class TriConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum')
        self.lin1 = Linear(in_channels,  out_channels)

    def forward(self, edge_index, r_matrix, f):
        maxi = self.propagate(edge_index, r_matrix=r_matrix, f=f.unsqueeze(1))
        w_phi_output = self.linear_phi(maxi)
        return w_phi_output

    def message(self, r_matrix, f_i, f_j):
        f_diff = f_i - f_j
        w_theta_output: torch.Tensor = self.linear_theta(x_i - x_j)
        return w_theta_output
