import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	mat_A = np.array(A, dtype=float)
	mat_T = np.array(T, dtype=float)
	mat_S = np.array(S, dtype=float)

	if np.isclose(np.linalg.det(mat_T), 0.0) or np.isclose(np.linalg.det(mat_S), 0.0):
		return -1
		

	T_inv = np.linalg.inv(mat_T)
	
	result = T_inv @ mat_A @ mat_S
	
	return result.tolist()