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

def predict_(x, theta):
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
	return np.dot(newX, theta).astype(float)

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
	TxCopy = xCopy.transpose()
	return (TxCopy @ (xCopy @ theta - y)) / x.shape[0]

def fit_(x, y, theta, alpha, max_iter):
	"""
	Description:
		Fits the model to the training dataset contained in x and y.
	Args:
		x: has to be a numpy.array, a matrix of dimension m * n:
		y: has to be a numpy.array, a vector of dimension m * 1:
		theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
		alpha: has to be a float, the learning rate
		max_iter: has to be an int, the number of iterations done during the gradient descent
	Return:
		new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
		None if there is a matching dimension problem.
		None if x, y, theta, alpha or max_iter is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, -1) or not check_matrix(y, x.shape[0], 1) or not check_matrix(theta, (x.shape[1] + 1), 1) or not isinstance(alpha, float) or not isinstance(max_iter, int) or max_iter < 0):
		return None
	thetaCopy = theta.copy()
	for i in range(max_iter):
		thetaCopy = thetaCopy - (alpha * gradient(x, y, thetaCopy))
	return thetaCopy
		
#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#


if __name__ == "__main__":
	x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
	y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
	theta = np.array([[42.], [1.], [1.], [1.]])
	theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
	ret2 = predict_(x, theta2)
	print(theta2)
	print(ret2)
	print("\n-------   Aproximative result   -------\n")
	print(np.array([[41.99],[0.97], [0.77], [-1.20]]))
	print(np.array([[19.5992], [-2.8003], [-25.1999], [-47.5996]]))

