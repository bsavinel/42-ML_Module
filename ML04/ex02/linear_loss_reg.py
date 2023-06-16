import numpy as np

#!########################################################################################################!#
#!##############################################  FUNCTION  ##############################################!#
#!########################################################################################################!#

def check_matrix(m, sizeX = -1, sizeY = -1, dim = 2):
	if (not isinstance(m, np.ndarray)):
		return False
	if (m.ndim != dim or m.size == 0):
		return False
	if (sizeX != -1 and m.shape[0] != sizeX):
		return False
	if (dim == 2 and sizeY != -1 and m.shape[1] != sizeY):
		return False
	return True

def l2(theta):
	"""Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
	Args:
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
	Returns:
		The L2 regularization as a float.
		None if theta in an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(theta, -1, 1)):
		return None
	newTheta = np.copy(theta).reshape(-1)
	newTheta[0] = 0
	return np.dot(newTheta, newTheta)

def reg_loss_(y, y_hat, theta, lambda_):
	"""Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.
	Args:
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
		lambda_: has to be a float.
	Returns:
		The regularized loss as a float.
		None if y, y_hat, or theta are empty numpy.ndarray.
		None if y and y_hat do not share the same shapes.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(y, -1, 1) or not check_matrix(y_hat, y.shape[0], 1) or not check_matrix(theta, -1, 1) or not isinstance(lambda_, float)):
		return None
	sub = y_hat.reshape(-1) - y.reshape(-1)
	return (np.dot(sub,sub) + (lambda_ * l2(theta))) / (2 * y.shape[0])

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
	theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
	print(reg_loss_(y, y_hat, theta, .5))
	print(reg_loss_(y, y_hat, theta, .05))
	print(reg_loss_(y, y_hat, theta, .9))
	print("\n-------   Result expected   -------\n")
	print(0.8503571428571429)
	print(0.5511071428571429)
	print(1.116357142857143)
