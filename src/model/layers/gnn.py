import torch

from model.layers.devconv_gnn import DevConvGNN

class GNN_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.devconv1 = DevConvGNN(3, 64, 3)
        self.devconv2 = DevConvGNN(3, 64, 3)
        self.devconv3 = DevConvGNN(3, 64, 1)
    

    def forward(self, graph):
        edge_index, x = graph.edge_index.long(), graph.x

        x = self.devconv1(edge_index, x)
        x = torch.relu(x)

        x = self.devconv2(edge_index, x)
        x = torch.relu(x)

        x = self.devconv3(edge_index, x)
        x = torch.sigmoid(x)
        return x