import numpy as np

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
	pred = xprime @ theta
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
	pred = Xprime @ theta
	return (np.matmul(Xprime.T,(pred - y)) + (lambda_ * newTheta)) / y.shape[0]


if __name__ == '__main__':
	x = np.array([[ -6, -7, -9],[ 13, -2, 14],[ -7, 14, -1],[ -8, -4, 6],[ -5, -9, 6],[ 1, -5, 11],[ 9, -11, 8]])
	y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
	theta = np.array([[7.01], [3], [10.5], [-6]])

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
	print(np.array([[ -60.99 ],[-195.64714286],[ 863.46571429],[-644.52142857]]))
	print(np.array([[ -60.99 ],[-195.64714286],[ 863.46571429],[-644.52142857]]))
	print(np.array([[ -60.99 ],[-195.86142857],[ 862.71571429],[-644.09285714]]))
	print(np.array([[ -60.99 ],[-195.86142857],[ 862.71571429],[-644.09285714]]))
	print(np.array([[ -60.99 ],[-196.07571429],[ 861.96571429],[-643.66428571]]))
	print(np.array([[ -60.99 ],[-196.07571429],[ 861.96571429],[-643.66428571]]))