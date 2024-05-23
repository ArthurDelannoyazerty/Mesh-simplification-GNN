import torch.nn as nn
from layers.devcon import DevConv

class GNN_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.devconv1 = DevConv(1)
        self.devconv2 = DevConv(64)
        self.devconv3 = DevConv(1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sigmoid3 = nn.Sigmoid()
    

    def forward(self, x, nodes, adjacency_matrix):
        x = self.devconv1(x, nodes, adjacency_matrix)
        x = self.relu1(x)
        x = self.devconv2(x, nodes, adjacency_matrix)
        x = self.relu2(x)
        x = self.devconv3(x, nodes, adjacency_matrix)
        x = self.sigmoid3(x)
        return x