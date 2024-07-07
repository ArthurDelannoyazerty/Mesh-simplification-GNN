import torch
import torch_cluster
from model.layers.knn import KNN

def triangle_generator_loss(original_nodes, original_barycenters, selected_triangles, selected_triangles_probabilities):
    
    b = original_barycenters
    b_hat = selected_triangles.mean(dim=1)
    p_b_hat = selected_triangles_probabilities

    d_f_S_Ss = _torch_d_f_S_Ss(p_b_hat, b_hat, b)
    d_r_S_Ss = _torch_d_r_S_Ss(original_nodes, selected_triangles, selected_triangles_probabilities, b_hat)

    return d_f_S_Ss, d_r_S_Ss




def _torch_d_f_S_Ss(p_b_hat, b_hat, b):
    distances = torch.cdist(b_hat,b)
    distances_filtered = torch.where(distances != 0, distances, torch.inf)
    min_b = distances_filtered.min(dim=1).values
    final_term = torch.sum(p_b_hat * min_b)
    return final_term


def _sample_points_in_triangles(triangles, num_points_per_triangle=50):
    """
    Sample points uniformly within N triangles in 3D space.
    
    Args:
    triangles (torch.Tensor): Tensor of triangle vertices, shape (N, 3, 3)
                              where N is the number of triangles
    num_points_per_triangle (int): Number of points to sample per triangle
    
    Returns:
    torch.Tensor: Sampled points, shape (N * num_points_per_triangle, 3)
    """
    N = triangles.shape[0]
    
    # Generate random barycentric coordinates
    r1 = torch.sqrt(torch.rand(N, num_points_per_triangle, device=torch.device('cuda')))
    r2 = torch.rand(N, num_points_per_triangle, device=torch.device('cuda'))
    
    # Convert barycentric coordinates to Cartesian coordinates
    a = 1 - r1
    b = r1 * (1 - r2)
    c = r1 * r2
    
    # Reshape for broadcasting
    a = a.unsqueeze(2)
    b = b.unsqueeze(2)
    c = c.unsqueeze(2)
    
    # Compute the sampled points
    sampled_points = (a * triangles[:, 0].unsqueeze(1) +
                      b * triangles[:, 1].unsqueeze(1) +
                      c * triangles[:, 2].unsqueeze(1))
    
    # Reshape to (N * num_points_per_triangle, 3)
    # sampled_points = sampled_points.view(-1, 3)
    
    return sampled_points



def _torch_d_r_S_Ss(original_nodes, generated_triangles, probability_generated_triangles, barycenters_generated_triangles):
    k = 20

    sampled_points_generated_triangles = _sample_points_in_triangles(generated_triangles)

    # First term
    distances = torch.cdist(original_nodes, sampled_points_generated_triangles)
    min_distances = distances.min(dim=1).values
    min_mean_distances = min_distances.mean(dim=1)
    first_term = probability_generated_triangles * min_mean_distances
    
    # Second term
    # Average distance to k-nearest points
    # row, col = torch_cluster.knn(barycenters_generated_triangles, barycenters_generated_triangles, k=k)
    
    # # Reshape indices to group by query point
    # row = row.view(-1, k)
    # col = col.view(-1, k)

    indices_knn = KNN(k=k, batch_size=1000)(barycenters_generated_triangles).int()
    
    # Compute distances, excluding self-connections
    x_tk = sampled_points_generated_triangles[indices_knn[:,1:]]                    # shape (nb_triangles_generated, nb_triangles_neighbors, nb_points_sample_per_triangle, 3)
    x_y  = sampled_points_generated_triangles[indices_knn[:,0]].unsqueeze(1)        # Represent respectfully the points on each neighbors triangles and the points on the current triangle 
    knn_distances = torch.cdist(x_tk, x_y)                                  # shape : (nb_triangles, nb_neighbors, nb_points_per_triangles, nb_points_per_triangles)
    
    knn_distances_mean = knn_distances.mean(dim=3)
    probabilities_tk = probability_generated_triangles[indices_knn[:,1:]]
    dist_times_probability = knn_distances_mean * probabilities_tk.unsqueeze(2)
    sum_over_k = dist_times_probability.sum(dim=1)
    mean_points_per_triangle_again = sum_over_k.mean(dim=1)
    normed = mean_points_per_triangle_again / k
    second_term = normed * (1-probability_generated_triangles)

    loss_d_r_S_Ss = torch.sum(first_term + second_term)
    
    return loss_d_r_S_Ss