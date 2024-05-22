import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, r_matrix, indices_neigh_tri, hidden_size):
    super().__init__()
    self.r_matrix = r_matrix
    self.indices_neigh_tri = indices_neigh_tri
    self.hidden_size = hidden_size

  def forward(self, p_init):
    neigh_all = self.indices_neigh_tri[:,1:]

    # Triconv 1
    f = p_init
    diff_p_all = (f.repeat((neigh_all.shape[1],1)).T - f[neigh_all])
    r_diff = torch.cat((self.r_matrix, diff_p_all.unsqueeze(-1)), dim=2)
    
    x = nn.Flatten()(r_diff)
    x = nn.Linear(r_diff.shape[1]*r_diff.shape[2], self.hidden_size)(x)
    x = nn.ReLU()(x)
    f = nn.Linear(self.hidden_size, 1)(x).squeeze()

    # Triconv 2
    diff_p_all = (f.repeat((neigh_all.shape[1],1)).T - f[neigh_all])
    r_diff = torch.cat((self.r_matrix, diff_p_all.unsqueeze(-1)), dim=2)

    x = nn.Flatten()(r_diff)
    x = nn.Linear(r_diff.shape[1]*r_diff.shape[2], self.hidden_size)(x)
    x = nn.ReLU()(x)
    f = nn.Linear(self.hidden_size, 1)(x).squeeze()

    # Triconv 3
    diff_p_all = (f.repeat((neigh_all.shape[1],1)).T - f[neigh_all])
    r_diff = torch.cat((self.r_matrix, diff_p_all.unsqueeze(-1)), dim=2)

    x = nn.Flatten()(r_diff)
    x = nn.Linear(r_diff.shape[1]*r_diff.shape[2], self.hidden_size)(x)
    x = nn.ReLU()(x)
    f = nn.Linear(self.hidden_size, 1)(x).squeeze()

    f_softmax = nn.Softmax()(f)
    
    return f_softmax