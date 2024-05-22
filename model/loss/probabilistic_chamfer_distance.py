import torch

def torch_d_P_Ps(p_y, x, y):
    """All Tensors in input"""
    # print(p_y.shape, x.shape, y.shape)

    expanded_x1 = x.unsqueeze(1)
    expanded_x2 = y.unsqueeze(0)
    distances = torch.norm(expanded_x1 - expanded_x2, dim=2)        # distance matrix

    min_x = distances.min(dim=1).values
    min_y = distances.min(dim=0)

    first_term = torch.sum(torch.index_select(p_y, 0, min_y.indices) * min_y.values)
    second_term = torch.sum(min_x * p_y)

    return first_term + second_term