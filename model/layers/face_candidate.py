import torch
import torch.nn as nn

class FaceCandidatesLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, S, adjacency_matrix):
        A_s = torch.matmul(torch.matmul(S, --adjacency_matrix), S.T)     # A_s = S * A * S.T
        A_s = A_s/A_s.max()                                                 # Normalize
        return A_s