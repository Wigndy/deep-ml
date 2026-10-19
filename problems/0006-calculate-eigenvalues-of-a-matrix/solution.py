def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	a, b = matrix[0][0], matrix[0][1]
	c, d = matrix[1][0], matrix[1][1]

	trace = a + d
	det = a* d - b * c
	delta = trace ** 2 -  4 * det

	lambda_1 = (trace + (delta ** 0.5)) / 2
	lambda_2 = (trace - (delta ** 0.5)) / 2

	eigenvalues = [lambda_1, lambda_2]
	eigenvalues.sort(reverse=True)

	return eigenvalues