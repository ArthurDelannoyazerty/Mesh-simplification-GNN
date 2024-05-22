import torch
import torch.nn as nn

class FirstPInitLayer(nn.Module):
    def __init__(self, A_s, triangles):
        super().__init__()
        self.A_s = A_s
        self.triangles = triangles

    def forward(self, triangles_indexes):
        # Extract indices for each triangle
        i, j, k = triangles_indexes.T

        # Extract probabilities using advanced indexing
        A_s_ij = self.A_s[i, j]
        A_s_ik = self.A_s[i, k]
        A_s_jk = self.A_s[j, k]

        # Calculate the barycenter probabilities
        p_init = torch.zeros(self.triangles.shape[0])
        p_init = (A_s_ij + A_s_ik + A_s_jk) / 3
        return p_init