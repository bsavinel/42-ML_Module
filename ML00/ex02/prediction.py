import numpy as np

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

def simple_predict(x, theta):
	"""Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
	Args:
		x: has to be an numpy.ndarray, a vector of dimension m * 1.
		theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
		y_hat as a numpy.ndarray, a vector of dimension m * 1.
		None if x or theta are empty numpy.ndarray.
		None if x or theta dimensions are not appropriate.
	Raises:
		This function should not raise any Exception.
	"""
	if ((not isinstance(x, np.ndarray)) or (not isinstance(theta, np.ndarray)) or x.size == 0 or theta.size != 2 or x.ndim != 1 or theta.ndim != 1):
		return None
	return np.array([float(theta[0] + theta[1] * x[i]) for i in range(x.shape[0])])

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	x = np.arange(1,6)
	# x = [1, 2, 3, 4, 5]
	theta1 = np.array([5, 0])
	theta2 = np.array([0, 1])
	theta3 = np.array([5, 3])
	theta4 = np.array([-3, 1])
	ret1 = simple_predict(x, theta1)
	ret2 = simple_predict(x, theta2)
	ret3 = simple_predict(x, theta3)
	ret4 = simple_predict(x, theta4)
	print(ret1)
	print(ret2)
	print(ret3)
	print(ret4)
	print("\n-------   Result expected   -------\n")
	print(np.array([5., 5., 5., 5., 5.]))
	print(np.array([1., 2., 3., 4., 5.]))
	print(np.array([ 8., 11., 14., 17., 20.]))
	print(np.array([-2., -1., 0., 1., 2.]))