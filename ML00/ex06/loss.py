import numpy as np

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

def predict_(x, theta):
	"""Computes the vector of prediction y_hat from two non-empty numpy.array.
	Args:
		x: has to be an numpy.array, a vector.
		theta: has to be an numpy.array, a vector of dimension 2 * 1.
	Returns:
		y_hat as a numpy.array, a vector of dimension m * 1.
		None if x and/or theta are not numpy.array.
		None if x or theta are empty numpy.array.
		None if x or theta dimensions are not appropriate.
	Raises:
		This function should not raise any Exceptions.
	"""
	if ((not isinstance(x, np.ndarray)) or (not isinstance(theta, np.ndarray))):
		return None
	copy = x.copy()
	copy.reshape(-1, 1)
	if ( copy.size == 0 or theta.size != 2 or copy.ndim != 2 or theta.ndim != 2):
		return None
	tmp = np.array([float(theta[0] + theta[1] * copy[i]) for i in range(copy.shape[0])])
	return tmp.reshape((tmp.shape[0], 1))

def loss_elem_(y, y_hat):
	"""
	Description:
		Calculates all the elements (y_pred - y)^2 of the loss function.
	Args:
		y: has to be an numpy.array, a vector.
		y_hat: has to be an numpy.array, a vector.
	Returns:
		J_elem: numpy.array, a vector of dimension (number of the training examples,1).
		None if there is a dimension matching problem between X, Y or theta.
		None if any argument is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if ((not isinstance(y, np.ndarray)) or (not isinstance(y_hat, np.ndarray))):
		return None
	copyY = y.copy()
	copyYHat = y_hat.copy()
	copyY.reshape(-1, 1)
	copyYHat.reshape(-1, 1)
	if (copyY.size == 0 or copyYHat.size == 0 or copyY.ndim != 2 or copyYHat.ndim != 2 or copyY.shape != copyYHat.shape):
		return None
	return np.array([(copyY[i] - copyYHat[i]) ** 2 for i in range(copyY.shape[0])])

def loss_(y, y_hat):
	"""
	Description:
		Calculates the value of loss function.
	Args:
		y: has to be an numpy.array, a vector.
		y_hat: has to be an numpy.array, a vector.
	Returns:
		J_value : has to be a float.
		None if there is a dimension matching problem between X, Y or theta.
		None if any argument is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	ret = loss_elem_(y, y_hat)
	if (ret is None):
		return None
	return np.sum(ret) / (2 * y.shape[0])

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	print("\nFirst test\n")

	x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
	theta1 = np.array([[2.], [4.]])
	y_hat1 = predict_(x1, theta1)
	y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

	ret1 = loss_elem_(y1, y_hat1)
	ret2 = loss_(y1, y_hat1)
	print(ret1)
	print(ret2)

	print("\nSecond test\n")

	x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
	theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
	y_hat2 = predict_(x2, theta2)
	y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)

	ret3 = loss_(y2, y_hat2)
	ret4 = loss_(y2, y2)
	print(ret3)
	print(ret4)
	print("\n-------   Result expected   -------\n")
	print("\nFirst test\n")
	print(np.array([[0.], [1], [4], [9], [16]]))
	print(3.0)
	print("\nSecond test\n")
	print(2.142857142857143)
	print(0.0)