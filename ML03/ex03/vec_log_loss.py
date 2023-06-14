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
	return 1 / (1 + np.exp(np.dot(copyx, theta) * -1))	


def vec_log_loss_(y, y_hat, eps=1e-15):
	"""
	Compute the logistic loss value.
	Args:
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
		eps: epsilon (default=1e-15)
	Returns:
		The logistic loss value as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(y, -1, 1) or not check_matrix(y_hat, y.shape[0], 1) or not isinstance(eps, float)):
		return None
	copyYHat = y_hat.copy()
	copyYHat = np.where(copyYHat == 1, copyYHat - eps, copyYHat)
	copyYHat = np.where(copyYHat == 0, copyYHat + eps, copyYHat)
	vec_one = np.ones(y.shape).reshape((-1, 1))
	dot1 = np.dot(y.reshape(-1), np.log(copyYHat).reshape(-1))
	dot2 = np.dot((vec_one - y).reshape(-1), np.log(vec_one - copyYHat).reshape(-1))
	return (dot1 + dot2) / (-y.shape[0])

	
#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	y1 = np.array([1]).reshape((-1, 1))
	x1 = np.array([4]).reshape((-1, 1))
	y2 = np.array([[1], [0], [1], [0], [1]])
	x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
	y3 = np.array([[0], [1], [1]])
	x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
	theta1 = np.array([[2], [0.5]])
	theta2 = np.array([[2], [0.5]])
	theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	y_hat1 = logistic_predict_(x1, theta1)
	y_hat2 = logistic_predict_(x2, theta2)
	y_hat3 = logistic_predict_(x3, theta3)
	ret1 = vec_log_loss_(y1, y_hat1)
	ret2 = vec_log_loss_(y2, y_hat2)
	ret3 = vec_log_loss_(y3, y_hat3)
	print(ret1)
	print(ret2)
	print(ret3)
	print("\n-------   Result expected   -------\n")
	print(0.018149927917808714)
	print(2.4825011602472347)
	print(2.993853310859968)