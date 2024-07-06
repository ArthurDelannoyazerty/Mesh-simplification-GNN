import torch
import torch.nn as nn

class FirstPInitLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, triangles_indexes, A_s, triangles):
        # Extract indices for each triangle
        i, j, k = triangles_indexes.T

        # Extract probabilities using advanced indexing
        A_s_ij = A_s[i, j]
        A_s_ik = A_s[i, k]
        A_s_jk = A_s[j, k]

        # Calculate the barycenter probabilities
        p_init = torch.zeros(triangles.shape[0])
        p_init = (A_s_ij + A_s_ik + A_s_jk) / 3
        return p_init