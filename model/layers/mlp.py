import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, hidden_size, k):
    super().__init__()
    self.k = k

    self.linear1 = nn.Linear(6, hidden_size)
    self.relu1 = nn.ReLU()

    self.linear2 = nn.Linear(6, hidden_size)
    self.relu2 = nn.ReLU()

    self.linear3 = nn.Linear(6, hidden_size)
    self.softmax = nn.Softmax(dim=0)


  def forward(self, p_init, r_matrix, indices_neigh_tri):
    neigh_all = indices_neigh_tri[:,1:]

    # Triconv 1
    f = p_init
    diff_p_all = (f.repeat((neigh_all.shape[1],1)).T - f[neigh_all])
    r_diff = torch.cat((r_matrix, diff_p_all.unsqueeze(-1)), dim=2)
    
    mlp_output = self.linear1(r_diff)
    f_output = mlp_output.sum(dim=1).sum(dim=1)   # sum over hidden dim and k 
    f = self.relu1(f_output)

    # Triconv 2
    diff_p_all = (f.repeat((neigh_all.shape[1],1)).T - f[neigh_all])
    r_diff = torch.cat((r_matrix, diff_p_all.unsqueeze(-1)), dim=2)
    
    mlp_output = self.linear2(r_diff)
    f_output = mlp_output.sum(dim=1).sum(dim=1)   # sum over hidden dim and k 
    f = self.relu1(f_output)

    # Triconv 3
    diff_p_all = (f.repeat((neigh_all.shape[1],1)).T - f[neigh_all])
    r_diff = torch.cat((r_matrix, diff_p_all.unsqueeze(-1)), dim=2)
    
    mlp_output = self.linear3(r_diff)
    f_output = mlp_output.sum(dim=1).sum(dim=1)   # sum over hidden dim and k 
    f_softmax = self.softmax(f_output)
    
    return f_softmax