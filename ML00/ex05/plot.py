import numpy as np
import matplotlib.pyplot as plt

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

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
	if (not check_matrix(x, -1, -1, 1) or not check_matrix(theta, 2, 1, 2)):
		return None
	tmp = np.array([float(theta[0] + theta[1] * x[i]) for i in range(x.shape[0])])
	return tmp

def plot(x, y, theta):
	"""Plot the data and prediction line from three non-empty numpy.array.
	Args:
		x: has to be an numpy.array, a vector of dimension m * 1.
		y: has to be an numpy.array, a vector of dimension m * 1.
		theta: has to be an numpy.array, a vector of dimension 2 * 1.
	Returns:
		Nothing.
	Raises:
		This function should not raise any Exceptions.
	"""
	if (not check_matrix(x, -1, -1, 1) or not check_matrix(y, x.shape[0], -1, 1) or not check_matrix(theta, 2, 1, 2)):
		return 
	plt.plot(x, y, 'bo')
	plt.plot(x, predict_(x, theta), 'r')
	plt.show()

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	x = np.arange(1,6)
	y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
	theta1 = np.array([[4.5],[-0.2]])
	theta2 = np.array([[-1.5],[2]])
	theta3 = np.array([[3],[0.3]])
	plot(x, y, theta1)
	plot(x, y, theta2)
	plot(x, y, theta3)