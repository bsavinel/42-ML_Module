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


def gradient(x, y, theta):
	"""Computes a gradient vector from three non-empty numpy.array, without any for-loop.
	The three arrays must have the compatible dimensions.
	Args:
		x: has to be an numpy.array, a matrix of dimension m * n.
		y: has to be an numpy.array, a vector of dimension m * 1.
		theta: has to be an numpy.array, a vector (n + 1) * 1.
	Return:
		The gradient as a numpy.array, a vector of dimensions n * 1,
		containg the result of the formula for all j.
		None if x, y, or theta are empty numpy.array.
		None if x, y and theta do not have compatible dimensions.
		None if x, y or theta is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, -1) or not check_matrix(y, x.shape[0], 1) or not check_matrix(theta, x.shape[1] + 1, 1)):
		return None
	xCopy = np.insert(x,0, 1, axis = 1)
	return (np.matmul(xCopy.transpose(), np.matmul(xCopy, theta) - y)) / x.shape[0]

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#


if __name__ == "__main__":
	x = np.array([[ -6, -7, -9],[ 13, -2, 14],[ -7, 14, -1],[ -8, -4, 6],[ -5, -9, 6],[ 1, -5, 11],[ 9, -11, 8]])
	y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	theta1 = np.array([0,3,0.5,-6]).reshape((-1, 1))
	theta2 = np.array([0,0,0,0]).reshape((-1, 1))
	ret1 = gradient(x, y, theta1)
	ret2 = gradient(x, y, theta2)
	print(ret1)
	print(ret2)
	print("\n-------   Result expected   -------\n")
	print(np.array([[ -33.71428571], [ -37.35714286], [183.14285714], [-393.]]))
	print(np.array([[ -0.71428571], [ 0.85714286], [23.28571429], [-26.42857143]]))

