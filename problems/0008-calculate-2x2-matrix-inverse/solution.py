import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    """
    Compute the inverse of a 2x2 matrix using PyTorch.
    
    Args:
        matrix: A 2x2 matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A 2x2 tensor containing the inverse, or None if the matrix is singular
    """
    m = torch.as_tensor(matrix, dtype=torch.float)
    a, b = m[0, 0], m[0, 1]
    c, d = m[1, 0], m[1, 1]
    det = a * d - b * c
    if torch.isclose(det, torch.tensor(0.0)):   return None
    new_m = torch.tensor([[d, -b], [-c, a]], dtype=torch.float32)
    scalar = 1 / det
    return new_m * scalar