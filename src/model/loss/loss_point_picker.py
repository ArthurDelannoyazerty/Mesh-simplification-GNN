import torch

def torch_d_P_Ps(p_y, x, y, simplification_rate):
    """All Tensors in input"""
    distances = torch.cdist(x,y)
    min_x = distances.min(dim=1).values
    min_y = distances.min(dim=0)

    first_term = torch.sum(torch.index_select(p_y, 0, min_y.indices) * min_y.values)
    second_term = torch.sum(min_x * p_y)

    d_p_ps = first_term + second_term
    return d_p_ps * simplification_rate