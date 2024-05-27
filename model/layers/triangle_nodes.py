import torch.nn as nn

class TriangleNodes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, triangles_indexes, nodes):
        return nodes[triangles_indexes]