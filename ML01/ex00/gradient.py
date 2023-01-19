import numpy as np

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

def isVector(x):
	if ((isinstance(x, np.ndarray)) and (x.ndim == 2 and x.shape[1] == 1)):
		return True
	return False

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

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))
	theta2 = np.array([1, -0.4]).reshape((-1, 1))
	theta1 = np.array([2, 0.7]).reshape((-1, 1))
	ret1 = simple_gradient(x, y, theta1)
	ret2 = simple_gradient(x, y, theta2)
	print(ret1)
	print(ret2)
	print("\n-------   Result expected   -------\n")
	print(np.array([[-19.0342574], [-586.66875564]]))
	print(np.array([[-57.86823748], [-2230.12297889]]))