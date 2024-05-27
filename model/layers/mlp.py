import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, hidden_size, k):
    super().__init__()

    self.flatten1 = nn.Flatten()
    self.linear1_1 = nn.Linear(k*5, hidden_size)
    self.relu1 = nn.ReLU()
    self.linear1_2 = nn.Linear(hidden_size, 1)

    self.flatten2 = nn.Flatten()
    self.linear2_1 =  nn.Linear(k*5, hidden_size)
    self.relu2 = nn.ReLU()
    self.linear2_2 = nn.Linear(hidden_size, 1)

    self.flatten3 = nn.Flatten()
    self.linear3_1 =  nn.Linear(k*5, hidden_size)
    self.relu3 = nn.ReLU()
    self.linear3_2 = nn.Linear(hidden_size, 1)

    self.softmax = nn.Softmax()


  def forward(self, p_init, r_matrix, indices_neigh_tri):
    neigh_all = indices_neigh_tri[:,1:]

    # Triconv 1
    f = p_init
    diff_p_all = (f.repeat((neigh_all.shape[1],1)).T - f[neigh_all])
    r_diff = torch.cat((r_matrix, diff_p_all.unsqueeze(-1)), dim=2)
    
    x = self.flatten1(r_diff)
    x = self.linear1_1(x)
    x = self.relu1(x)
    f = self.linear1_2(x).squeeze()

    # Triconv 2
    diff_p_all = (f.repeat((neigh_all.shape[1],1)).T - f[neigh_all])
    r_diff = torch.cat((r_matrix, diff_p_all.unsqueeze(-1)), dim=2)

    x = self.flatten2(r_diff)
    x = nn.Linear(r_diff.shape[1]*r_diff.shape[2], self.hidden_size)(x)
    x = self.relu2(x)
    f = self.linear2_2(x).squeeze()

    # Triconv 3
    diff_p_all = (f.repeat((neigh_all.shape[1],1)).T - f[neigh_all])
    r_diff = torch.cat((r_matrix, diff_p_all.unsqueeze(-1)), dim=2)

    x = self.flatten3(r_diff)
    x = nn.Linear(r_diff.shape[1]*r_diff.shape[2], self.hidden_size)(x)
    x = self.relu3(x)
    f = self.linear3_2(x).squeeze()

    f_softmax = self.softmax(f)
    
    return f_softmax