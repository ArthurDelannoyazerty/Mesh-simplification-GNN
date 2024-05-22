import torch.nn as nn

class TriangleNodes(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = nodes

    def forward(self, triangles_indexes):
        return self.nodes[triangles_indexes]