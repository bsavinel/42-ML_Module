import numpy as np

#!########################################################################################################!#
#!##############################################  FUNCTION  ##############################################!#
#!########################################################################################################!#

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


def loss_(y, y_hat):
	"""Computes the mean squared error of two non-empty numpy.array, without any for loop.
	The two arrays must have the same dimensions.
	Args:
		y: has to be an numpy.array, a vector.
		y_hat: has to be an numpy.array, a vector.
	Return:
		The mean squared error of the two vectors as a float.
		None if y or y_hat are empty numpy.array.
		None if y and y_hat does not share the same dimensions.
		None if y or y_hat is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if ((not check_matrix(y, -1, -1, 1) and not check_matrix(y, -1, 1)) or not isinstance(y_hat, np.ndarray)):
		return None
	yCopy = y.copy().reshape(-1)
	y_hatCopy = y_hat.copy().reshape(-1)
	if (yCopy.shape != y_hatCopy.shape):
		return None
	diff = y_hatCopy - yCopy
	return np.dot(diff.T, diff) / (2 * yCopy.shape[0])

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#


if __name__ == "__main__":
	X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
	Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	ret1 = loss_(X, Y)
	ret2 = loss_(X, X)
	print(ret1)
	print(ret2)
	print("\n-------   Result expected   -------\n")
	print(2.142857142857143)
	print(0.0)
