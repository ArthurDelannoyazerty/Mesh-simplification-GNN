import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, input_channels, hidden_channels, output_channels):
    super().__init__()
    self.r_matrix_size = 5

    self.linear1 = nn.Linear(input_channels + self.r_matrix_size, hidden_channels)
    self.linear2 = nn.Linear(hidden_channels + self.r_matrix_size, hidden_channels)
    self.linear3 = nn.Linear(hidden_channels + self.r_matrix_size, output_channels)
    self.softmax = nn.Softmax(dim=0)


  def forward(self, p_init, r_matrix, indices_neigh_tri):
    neigh_all = indices_neigh_tri[:,1:]
    f = p_init.unsqueeze(1)

    # Triconv 1
    diff_p_all = f.unsqueeze(1) - f[neigh_all]
    r_diff = torch.cat((r_matrix, diff_p_all), dim=2)
    
    lin1_output = self.linear1(r_diff)
    f = lin1_output.sum(dim=1).relu()

    # Triconv 2
    diff_p_all = f.unsqueeze(1) - f[neigh_all]
    r_diff = torch.cat((r_matrix, diff_p_all), dim=2)
    
    lin2_output = self.linear2(r_diff)
    f = lin2_output.sum(dim=1).relu()

    # Triconv 3
    diff_p_all = f.unsqueeze(1) - f[neigh_all]
    r_diff = torch.cat((r_matrix, diff_p_all), dim=2)
    
    lin3_output = self.linear3(r_diff)
    f = self.softmax(lin3_output.sum(dim=1).squeeze())

    return f