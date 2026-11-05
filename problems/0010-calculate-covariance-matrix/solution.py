import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float)
    # Your implementation here
    mean_t = torch.mean(v_t, dim=1, keepdim=True)
    new_t = v_t - mean_t
    n_observations = v_t.shape[1]
    covariance_matrix = torch.matmul(new_t, new_t.T) / (n_observations - 1)
    
    return covariance_matrix