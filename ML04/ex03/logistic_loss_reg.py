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

def reg_log_loss_(y, y_hat, theta, lambda_):
	"""Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for lArgs:
	Args:
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
		lambda_: has to be a float.
	Returns:
		The regularized loss as a float.
		None if y, y_hat, or theta is empty numpy.ndarray.
		None if y and y_hat do not share the same shapes.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(y, -1, 1) or not check_matrix(y_hat, y.shape[0], 1) or not check_matrix(theta, -1, 1) or not isinstance(lambda_, float)):
		return None
	return vec_log_loss_(y, y_hat) + lambda_ / (2 * y.shape[0]) * l2(theta)
	
#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
	y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
	theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
	ret1 = reg_log_loss_(y, y_hat, theta, .5)
	ret2 = reg_log_loss_(y, y_hat, theta, .05)
	ret3 = reg_log_loss_(y, y_hat, theta, .9)
	print(ret1)
	print(ret2)
	print(ret3)
	print("\n-------   Result expected   -------\n")
	print(0.43377043716475955)
	print(0.13452043716475953)
	print(0.6997704371647596)