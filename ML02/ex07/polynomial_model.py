import numpy as np

def check_matrix(m, sizeX, sizeY, dim = 2):
	"""Check if the matrix corectly match the expected dimension.
	Args:
		m: the element to check.
		sizeX: the number of row, if sizeX = -1 isn't check that.
		sizeY: the number of collum, if sizeX = -1 isn't check that.
		dim: the dimension of the matrix. (only 2(default) or 1)
	Return:
		True if the matrix match the expected dimension.
		False if the matrix doesn't match the expected dimension or isn't a np.ndarray.
	"""
	if (not isinstance(m, np.ndarray)):
		return False
	if (m.ndim != dim or m.size == 0):
		return False
	if (sizeX != -1 and m.shape[0] != sizeX):
		return False
	if (dim == 2 and sizeY != -1 and m.shape[1] != sizeY):
		return False
	return True

def add_polynomial_features(x, power):
	"""Add polynomial features to vector x by raising its values up to the power given in argument.
	Args:
		x: has to be an numpy.array, a vector of dimension m * 1.
		power: has to be an int, the power up to which the components of vector x are going to be raised.
	Return:
		The matrix of polynomial features as a numpy.array, of dimension m * n,
		containing the polynomial feature values for all training examples.
		None if x is an empty numpy.array.
		None if x or power is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, 1) or not isinstance(power, int)):
		return None
	newx = np.repeat(x, power, axis=1)
	puiss = np.arange(1, power + 1)
	return np.power(newx, puiss)


if __name__ == "__main__":
	x = np.arange(1,6).reshape(-1, 1)
	ret1 = add_polynomial_features(x, 3)
	ret2 = add_polynomial_features(x, 6)
	print(ret1)
	print(ret2)
	print(np.array([[ 1, 1, 1],[ 2, 4, 8],[ 3, 9, 27],[ 4, 16, 64],[ 5, 25, 125]]))
	print(np.array([[ 1, 1, 1, 1, 1, 1],[ 2, 4, 8, 16, 32, 64],[ 3, 9, 27, 81, 243, 729],[ 4, 16, 64, 256, 1024, 4096],[ 5, 25, 125, 625, 3125, 15625]]))