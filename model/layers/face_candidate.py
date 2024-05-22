import torch
import torch.nn as nn

class FaceCandidatesLayer(nn.Module):
    def __init__(self, adjacency_matrix):
        super().__init__()
        self.adjacency_matrix = adjacency_matrix

    def forward(self, S):
        A_s = torch.matmul(torch.matmul(S, self.adjacency_matrix), S.T)     # A_s = S * A * S.T
        A_s = A_s/A_s.max()                                                 # Normalize
        return A_s