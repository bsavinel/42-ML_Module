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
	"""Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power give
	Args:
		x: has to be an numpy.ndarray, a matrix of shape m * n.
		power: has to be an int, the power up to which the columns of matrix x are going to be raised.
	Returns:
		The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature va
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, -1) or not isinstance(power, int)):
		return None
	newx = np.copy(x)
	for i in range(power - 1):
		newx = np.concatenate([x, newx], axis=1)
	puiss = np.arange(1, power + 1)
	puiss = np.repeat(puiss, x.shape[1]).reshape((1, -1))
	pui = np.copy(puiss)
	for i in range(x.shape[0] - 1):
		pui = np.concatenate([pui, puiss])
	return np.power(newx, pui)

if __name__ == "__main__":
	x = np.arange(1,11).reshape(5, 2)
	print(add_polynomial_features(x, 3))
	print(add_polynomial_features(x, 4))
	print("\n-------   Result expected   -------\n")
	print(np.array([[1,2,1,4,1,8],
    		[3,4,9,16,27,64],
	   		[5,6,25,36,125,216],
			[7,8,49,64,343,512],
    		[9,10,81, 100, 729, 1000]]))
	print(np.array([[1,2,1,4,1,8,1,16],
    		[3,4,9,16,27,64,81,256],
    		[5,6,25,36,125,216,625,1296],
    		[7,8,49,64,343,512,2401,4096],
    		[9,10,81, 100, 729, 1000,6561,10000]]))