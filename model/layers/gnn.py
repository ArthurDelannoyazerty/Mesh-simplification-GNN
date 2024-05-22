import torch.nn as nn
from layers.devcon import DevConv

class GNN_Model(nn.Module):
    def __init__(self, nodes, adjacency_matrix):
        super(GNN_Model, self).__init__()
        self.devconv = DevConv(nodes, adjacency_matrix, 1)
        self.relu = nn.ReLU()
        self.devconv2 = DevConv(nodes, adjacency_matrix, 64)
        self.relu2 = nn.ReLU()
        self.devconv3 = DevConv(nodes, adjacency_matrix,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x= self.devconv(x)
        x= self.relu(x)
        x= self.devconv2(x)
        x= self.relu2(x)
        x= self.devconv3(x)
        x= self.sigmoid(x)
        return x