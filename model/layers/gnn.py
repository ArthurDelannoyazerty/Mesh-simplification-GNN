import torch.nn as nn
import time
from layers.devconv_gnn import DevConvGNN

class GNN_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.devconv1 = DevConvGNN(3, 64, 3)
        self.devconv2 = DevConvGNN(3, 64, 3)
        self.devconv3 = DevConvGNN(3, 64, 1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sigmoid3 = nn.Sigmoid()
    

    def forward(self, graph):
        
        edge_index, x = graph.edge_index.long(), graph.x

        start = time.time()
        x = self.devconv1(edge_index, x)
        end = time.time()
        print('gnn_model l1 : ', end - start)
        
        x = self.relu1(x)
        start = time.time()
        x = self.devconv2(edge_index, x)
        end = time.time()
        print('gnn_model l2 : ', end - start)
        x = self.relu2(x)

        start = time.time()
        x = self.devconv3(edge_index, x)
        end = time.time()
        print('gnn_model l3 : ', end - start)
        x = self.sigmoid3(x)
        return x