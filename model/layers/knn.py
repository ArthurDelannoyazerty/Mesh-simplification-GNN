import torch
import torch.nn as nn

class KNN(nn.Module):
    def __init__(self, barycenters):
        super().__init__()
        self.barycenters = barycenters

    def forward(self, x, k=20, batch_size=100):
        indices_knn = torch.empty(size=(x.shape[0], k))
        
        modulo = x.shape[0]%batch_size
        nb_iter = int((x.shape[0] - modulo) / batch_size)

        for i in range(nb_iter):
            i_start, i_end = i*batch_size, (i+1)*batch_size
            distances = torch.norm(self.barycenters[i_start:i_end].unsqueeze(1) - self.barycenters.unsqueeze(0), dim=2)

            neighbors = distances.topk(k, dim=1, largest=False).indices.clone()  # Indices of the k-nearest neighbors
            indices_knn[i_start:i_end] = neighbors

        # last piece of computation
        distances = torch.norm(self.barycenters[-modulo:].unsqueeze(1) - self.barycenters.unsqueeze(0), dim=2)
        neighbors = distances.topk(k, dim=1, largest=False).indices.clone()  # Indices of the k-nearest neighbors
        indices_knn[-modulo:] = neighbors


        return indices_knn