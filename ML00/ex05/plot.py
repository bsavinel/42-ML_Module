import numpy as np
import matplotlib.pyplot as plt

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

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
	if ((not isinstance(x, np.ndarray)) or (not isinstance(theta, np.ndarray))  or x.size == 0 or theta.size != 2 or x.ndim != 1 or theta.ndim != 2):
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
	if (x.size == 0 or y.size == 0 or theta.size == 0 or x.ndim != 1 or y.ndim != 1 or theta.ndim != 2 or x.shape[0] != y.shape[0] or theta.shape[0] != 2 or theta.shape[1] != 1):
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