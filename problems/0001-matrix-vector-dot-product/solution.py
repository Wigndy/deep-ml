def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	m = len(a)
	n = len(a[0])

	m_b = len(b)
	if n != m_b:
		return -1
	aggre = lambda row: sum(map(lambda a, b: a * b, row, b))
	return list(map(aggre, a))