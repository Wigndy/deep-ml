import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    B = A.T @ A
    B11, B12 = B[0, 0], B[0, 1]
    B21, B22 = B[1, 0], B[1, 1]

    if np.isclose(B11, B22):
        theta = np.pi / 4
    else:
        theta = 0.5 * np.arctan2(2*B12, B11 - B22)

    c, s = np.cos(theta), np.sin(theta)

    R = np.array([[c, -s],
                  [s,  c]])

    V = R
    Vt = V.T
    
    D = Vt @ B @ V
    lam1, lam2 = D[0, 0], D[1, 1]
    
    lam1 = max(0.0, lam1)
    lam2 = max(0.0, lam2)
    s1 = np.sqrt(lam1)
    s2 = np.sqrt(lam2)
    
    if s1 < s2:
        s1, s2 = s2, s1
        Vt = Vt[[1, 0], :]
        
    S = np.array([s1, s2])

    v1 = Vt[0, :].reshape(-1, 1)
    v2 = Vt[1, :].reshape(-1, 1)
    
    if s1 > 1e-9:
        u1 = (A @ v1) / s1
    else:
        u1 = np.array([[1.0], [0.0]])

    if s2 > 1e-9:
        u2 = (A @ v2) / s2
    else:
        u2 = np.array([[-u1[1, 0]], [u1[0, 0]]])

    U = np.hstack((u1,u2))

    return U, S, Vt