import numpy as np

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

def isVector(x):
	if ((isinstance(x, np.ndarray)) and (x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1))):
		return True
	return False

def loss_(y, y_hat):
	"""Computes the half mean squared error of two non-empty numpy.array, without any for loop.
	The two arrays must have the same dimensions.
	Args:
		y: has to be an numpy.array, a vector.
		y_hat: has to be an numpy.array, a vector.
	Returns:
		The half mean squared error of the two vectors as a float.
		None if y or y_hat are empty numpy.array.
		None if y and y_hat does not share the same dimensions.
	Raises:
		This function should not raise any Exceptions.
	"""
	if ((not isVector(y)) or (not isVector(y_hat))):
		return None
	copyY = y.copy()
	copyYHat = y_hat.copy()
	copyY.reshape(-1, 1)
	copyYHat.reshape(-1, 1)
	if (copyY.size == 0 or copyYHat.size == 0 or copyY.ndim != 2 or copyYHat.ndim != 2 or copyY.shape != copyYHat.shape):
		return None
	ret = (copyYHat - copyY)
	# dot c'est une multiplication matriciel donc faut transposer
	return np.dot(ret.T, ret)[0][0] / (2 * copyY.shape[0])

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
	Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
	ret1 = loss_(X, Y)
	ret2 = loss_(X, X)
	print(ret1)
	print(ret2)
	print("\n-------   Result expected   -------\n")
	print(2.142857142857143)
	print(0.0)