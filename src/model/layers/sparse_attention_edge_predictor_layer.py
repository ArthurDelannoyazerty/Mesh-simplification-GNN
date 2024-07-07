import torch
import torch.nn as nn

class SparseAttentionEdgePredictorLayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.wq = nn.Parameter(torch.Tensor(size))
        self.wk = nn.Parameter(torch.Tensor(size))
        nn.init.normal_(self.wq)
        nn.init.normal_(self.wk)


    def forward(self, f, neighbors):
        wq_f = self.wq.reshape(-1, 1) * f               # Wq*f
        wk_f = self.wk.reshape(-1, 1) * f               # Wq*f
        S = torch.exp(torch.matmul(wq_f.T, wk_f))       # e^((wq_f.T)*(wk_f))
        S = S / torch.matmul(neighbors, S).T    # Division
        return S