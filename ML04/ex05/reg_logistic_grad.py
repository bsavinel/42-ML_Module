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

def reg_linear_grad(y, x, theta, lambda_):
	"""Computes the regularized linear gradient of three non-empty numpy.ndarray,
		with two for-loop. The three arrays must have compatible shapes.
	Args:
		y: has to be a numpy.ndarray, a vector of shape m * 1.
		x: has to be a numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
		lambda_: has to be a float.
	Return:
		A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles shapes.
		None if y, x or theta or lambda_ is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, -1) or not check_matrix(y, x.shape[0], 1) or not check_matrix(theta, x.shape[1] + 1, 1) or not isinstance(lambda_, float)):
		return None
	xprime = np.insert(x, 0, 1, axis=1)
	pred = sigmoid_(xprime @ theta)
	tmp = pred - y
	newTheta = np.copy(theta).reshape(-1,1)
	newTheta[0][0] = 0
	return np.array([(sum(tmp * xprime[:, i].reshape(-1,1)) + (newTheta[i][0] * lambda_)) / y.shape[0] for i in range(theta.shape[0])])

def vec_reg_linear_grad(y, x, theta, lambda_):
	"""Computes the regularized linear gradient of three non-empty numpy.ndarray,
		without any for-loop. The three arrays must have compatible shapes.
	Args:
		y: has to be a numpy.ndarray, a vector of shape m * 1.
		x: has to be a numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
		lambda_: has to be a float.
	Return:
		A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles shapes.
		None if y, x or theta or lambda_ is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, -1) or not check_matrix(y, x.shape[0], 1) or not check_matrix(theta, x.shape[1] + 1, 1) or not isinstance(lambda_, float)):
		return None
	newTheta = np.copy(theta).reshape(-1,1)
	newTheta[0][0] = 0
	Xprime = np.insert(x, 0, 1, axis=1)
	pred = sigmoid_(Xprime @ theta)
	return (np.matmul(Xprime.T,(pred - y)) + (lambda_ * newTheta)) / y.shape[0]

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	x = np.array([[0, 2, 3, 4],[2, 4, 5, 5],[1, 3, 2, 7]])
	y = np.array([[0], [1], [1]])
	theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

	ret1 = reg_linear_grad(y, x, theta, 1.)
	ret2 = vec_reg_linear_grad(y, x, theta, 1.)
	ret3 = reg_linear_grad(y, x, theta, 0.5)
	ret4 = vec_reg_linear_grad(y, x, theta, 0.5)
	ret5 = reg_linear_grad(y, x, theta, 0.0)
	ret6 = vec_reg_linear_grad(y, x, theta, 0.0)
	print(ret1)
	print(ret2)
	print(ret3)
	print(ret4)
	print(ret5)
	print(ret6)
	print("\n-------   Result expected   -------\n")
	print(np.array([[-0.55711039],[-1.40334809],[-1.91756886],[-2.56737958],[-3.03924017]]))
	print(np.array([[-0.55711039],[-1.40334809],[-1.91756886],[-2.56737958],[-3.03924017]]))
	print(np.array([[-0.55711039],[-1.15334809],[-1.96756886],[-2.33404624],[-3.15590684]]))
	print(np.array([[-0.55711039],[-1.15334809],[-1.96756886],[-2.33404624],[-3.15590684]]))
	print(np.array([[-0.55711039],[-0.90334809],[-2.01756886],[-2.10071291],[-3.27257351]]))
	print(np.array([[-0.55711039],[-0.90334809],[-2.01756886],[-2.10071291],[-3.27257351]]))