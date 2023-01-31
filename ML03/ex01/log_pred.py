import numpy as np

#!########################################################################################################!#
#!##############################################  FUNCTION  ##############################################!#
#!########################################################################################################!#


def check_matrix(m, sizeX, sizeY, dim = 2):
	if (not isinstance(m, np.ndarray)):
		return False
	if (m.ndim != dim or m.size == 0):
		return False
	if (sizeX != -1 and m.shape[0] != sizeX):
		return False
	if (dim == 2 and sizeY != -1 and m.shape[1] != sizeY):
		return False
	return True

def sigmoid_(x):
	"""
	Compute the sigmoid of a vector.
	Args:
		x: has to be a numpy.ndarray of shape (m, 1).
	Returns:
		The sigmoid value as a numpy.ndarray of shape (m, 1).
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, 1)):
		return None
	return 1 / (1 + np.exp(x * -1))	

def logistic_predict_(x, theta):
	"""Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
	Args:
		x: has to be an numpy.ndarray, a vector of dimension m * n.
		theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
	Returns:
		y_hat as a numpy.ndarray, a vector of dimension m * 1.
		None if x or theta are empty numpy.ndarray.
		None if x or theta dimensions are not appropriate.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, -1) or not check_matrix(theta, x.shape[1] + 1, 1)):
		return None
	copyx = np.insert(x, 0, 1, axis = 1)
	return sigmoid_(np.dot(copyx, theta))

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	x = np.array([4]).reshape((-1, 1))
	theta = np.array([[2], [0.5]])
	ret1 = logistic_predict_(x, theta)
	x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
	theta2 = np.array([[2], [0.5]])
	ret2 = logistic_predict_(x2, theta2)
	x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
	theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	ret3 = logistic_predict_(x3, theta3)
	print(ret1)
	print(ret2)
	print(ret3)
	print("\n-------   Result expected   -------\n")
	print(np.array([[0.98201379]]))
	print(np.array([[0.98201379],[0.99624161],[0.97340301],[0.99875204],[0.90720705]]))
	print(np.array([[0.03916572],[0.00045262],[0.2890505 ]]))
	