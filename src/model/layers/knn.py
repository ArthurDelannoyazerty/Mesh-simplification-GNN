import torch
import torch.nn as nn

class KNN(nn.Module):
    def __init__(self, k, batch_size):
        super().__init__()
        self.k = k
        self.batch_size = batch_size

    def forward(self, barycenters):
        indices_knn = torch.empty(size=(barycenters.shape[0], self.k), device=torch.device('cuda'))
        
        modulo = barycenters.shape[0]%self.batch_size
        nb_iter = int((barycenters.shape[0] - modulo) / self.batch_size)

        for i in range(nb_iter):
            i_start, i_end = i*self.batch_size, (i+1)*self.batch_size
            distances = torch.cdist(barycenters[i_start:i_end],barycenters)
            neighbors = distances.topk(self.k, dim=1, largest=False).indices  # Indices of the k-nearest neighbors
            indices_knn[i_start:i_end] = neighbors

        if modulo!=0:
            # last piece of computation
            distances = torch.cdist(barycenters[-modulo:],barycenters)
            neighbors = distances.topk(self.k, dim=1, largest=False).indices  # Indices of the k-nearest neighbors
            indices_knn[-modulo:] = neighbors

        return indices_knn