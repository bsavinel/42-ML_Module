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

def plot_with_loss(x, y, theta):
	"""Plot the data and prediction line from three non-empty numpy.ndarray.
	Args:
		x: has to be an numpy.ndarray, a vector of dimension m * 1.
		y: has to be an numpy.ndarray, a vector of dimension m * 1.
		theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
		Nothing.
	Raises:
		This function should not raise any Exception.
	"""
	if ((not isinstance(x, np.ndarray)) or (not isinstance(y, np.ndarray)) or (not isinstance(theta, np.ndarray)) or x.size == 0 or y.size == 0 or theta.size == 0 or x.ndim != 1 or y.ndim != 1 or theta.ndim != 2 or x.shape[0] != y.shape[0] or theta.shape[0] != 2 or theta.shape[1] != 1):
		return
	plt.plot(x, y, 'bo')
	ret = predict_(x, theta)
	plt.plot(x, ret, 'r')
	for i in range(x.shape[0]):
		plt.plot([x[i],x[i]], [y[i],ret[i]], 'r--')
	plt.show()

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	x = np.arange(1,6)
	y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
	theta1 = np.array([[18],[-1]])
	theta2 = np.array([[14], [0]])
	theta3 = np.array([[12], [0.8]])
	plot_with_loss(x, y, theta1)
	plot_with_loss(x, y, theta2)
	plot_with_loss(x, y, theta3)
