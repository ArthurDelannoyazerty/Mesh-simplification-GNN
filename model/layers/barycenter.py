import torch.nn as nn

class BarycentersLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, triangles):
        return triangles.mean(1)