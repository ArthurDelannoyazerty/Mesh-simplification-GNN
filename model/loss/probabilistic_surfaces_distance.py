import torch

def torch_d_f_S_Ss(p_b_hat, b_hat, b):
    # print(p_b_hat.shape, b_hat.shape, b.shape)

    expanded_x1 = b_hat.unsqueeze(1)
    expanded_x2 = b.unsqueeze(0)
    distances = torch.norm(expanded_x1 - expanded_x2, dim=2)

    distances_filtered = torch.where(distances != 0, distances, torch.inf)

    min_b = distances_filtered.min(dim=1).values

    final_term = torch.sum(p_b_hat * min_b)

    return final_term


def torch_d_r_S_Ss(p_x, p_y, x ,y):
    expanded_x = x.unsqueeze(1)
    expanded_y = y.unsqueeze(0)
    distances = torch.norm(expanded_x - expanded_y, dim=2)
    min_d = distances.min(dim=0).values
    first_term = p_y * min_d


    indices_knn = distances.topk(k=50, dim=0, largest=False).indices.T  # Indices of the k-nearest neighbors
    knn_labels = x[indices_knn]
    xtk = torch.reshape(knn_labels, shape=((y.shape[0])*50, 3))

    expanded_xtk = xtk.unsqueeze(1)
    distances_knn = torch.norm(expanded_xtk - expanded_y, dim=2)
    min_knn = distances_knn.min(dim=1).values
    min_knn_reshaped = min_knn.reshape(((y.shape[0]), 50))

    ptk_time_norm = p_x[indices_knn] * min_knn_reshaped
    factor = (1-p_y) * (1/50)
    second_term = factor * torch.sum(ptk_time_norm, dim=1)

    final_term = torch.sum(first_term + second_term)

    return final_term