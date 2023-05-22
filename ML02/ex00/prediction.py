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


def simple_predict(x, theta):
	"""Computes the prediction vector y_hat from two non-empty numpy.array.
	Args:
		x: has to be an numpy.array, a matrix of dimension m * n.
		theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
	Return:
		y_hat as a numpy.array, a vector of dimension m * 1.
		None if x or theta are empty numpy.array.
		None if x or theta dimensions are not matching.
		None if x or theta is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, -1) or not check_matrix(theta, x.shape[1] + 1, 1)):
		return None
	newX = np.insert(x, 0, 1, axis = 1) #add 1 to the first collum to have the first theta value as constant
	tmpTheta = theta.reshape((-1,))
	return np.array([np.sum(newX[i] * tmpTheta) for i in range(newX.shape[0])]).reshape((-1, 1))

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#


if __name__ == "__main__":
	x = np.arange(1,13).reshape((4,-1))
	theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
	theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
	theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
	theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
	ret1 = simple_predict(x, theta1)
	ret2 = simple_predict(x, theta2)
	ret3 = simple_predict(x, theta3)
	ret4 = simple_predict(x, theta4)
	print(ret1)
	print(ret2)
	print(ret3)
	print(ret4)
	print("\n-------   Result expected   -------\n")
	print(np.array([[5.], [5.], [5.], [5.]]))
	print(np.array([[ 1.], [ 4.], [ 7.], [10.]]))
	print(np.array([[ 9.64], [24.28], [38.92], [53.56]]))
	print(np.array([[12.5], [32. ], [51.5], [71. ]]))