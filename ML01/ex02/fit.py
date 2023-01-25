import numpy as np

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

def isVector(x):
	if ((isinstance(x, np.ndarray)) and (x.ndim == 2 and x.shape[1] == 1)):
		return True
	return False

def predict(x, theta):
	"""Computes the vector of prediction y_hat from two non-empty numpy.array.
	Args:
		x: has to be an numpy.array, a vector of dimension m * 1.
		theta: has to be an numpy.array, a vector of dimension 2 * 1.
	Returns:
		y_hat as a numpy.array, a vector of dimension m * 1.
		None if x and/or theta are not numpy.array.
		None if x or theta are empty numpy.array.
		None if x or theta dimensions are not appropriate.
	Raises:
		This function should not raise any Exceptions.
	"""
	if ((not isinstance(x, np.ndarray)) or (not isinstance(theta, np.ndarray)) or x.size == 0 or theta.size != 2 or x.shape[1] != 1 or theta.ndim != 2):
		return None
	copyX = x.reshape(-1)
	tmp = np.array([float(theta[0] + theta[1] * copyX[i]) for i in range(copyX.shape[0])])
	return tmp.reshape((tmp.shape[0], 1))

def simple_gradient(x, y, theta):
	""" Computes a gradient vector from three non-empty numpy.array, with a for-loop.
		The three arrays must have compatible shapes.
	Args:
		x: has to be an numpy.array, a vector of shape m * 1.
		y: has to be an numpy.array, a vector of shape m * 1.
		theta: has to be an numpy.array, a 2 * 1 vector.
	Return:
		The gradient as a numpy.array, a vector of shape 2 * 1.
		None if x, y, or theta are empty numpy.array.
		None if x, y and theta do not have compatible shapes.
		None if x, y or theta is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if ((not isVector(x)) or (not isVector(y)) or y.shape[0] != x.shape[0] or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1):
		return None
	nabla0 = float((sum(((theta[1] * x) + theta[0]) - y)) / x.shape[0])
	nabla1 = float((sum((((theta[1] * x) + theta[0]) - y) * x)) / x.shape[0])
	return np.array([[nabla0], [nabla1]])

def fit_(x, y, theta, alpha, max_iter):
	"""
	Description:
		Fits the model to the training dataset contained in x and y.
	Args:
		x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
		y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
		theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
		alpha: has to be a float, the learning rate
		max_iter: has to be an int, the number of iterations done during the gradient descent
	Returns:
		new_theta: numpy.ndarray, a vector of dimension 2 * 1.
		None if there is a matching dimension problem.
	Raises:
		This function should not raise any Exception.
	"""
	if ((not isVector(x)) or (not isVector(y)) or (not isinstance(alpha, float)) or (not isinstance(max_iter, int)) or max_iter < 0 or y.shape[0] != x.shape[0] or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1):
		return None
	copyTheta = theta.copy()
	for i in range(max_iter):
		copyTheta = copyTheta - (alpha * simple_gradient(x, y, copyTheta))
	return copyTheta

	

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	theta= np.array([1, 1]).reshape((-1, 1))
	theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
	ret1 = predict(x, theta1)
	print(theta1)
	print(ret1)
	print("\n-------   Result expected   -------\n")
	print(np.array([[1.40709365],[1.1150909 ]]))
	print(np.array([[15.3408728 ],[25.38243697],[36.59126492],[55.95130097],[65.53471499]]))
