import numpy as np

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

def isVector(x):
	if ((isinstance(x, np.ndarray)) and (x.ndim == 2 and x.shape[1] == 1)):
		return True
	return False

def simple_gradient(x, y, theta):
	""" Computes a gradient vector from three non-empty numpy.array, without any for loop.
		The three arrays must have compatible shapes.
	Args:
		x: has to be a numpy.array, a matrix of shape m * 1.
		y: has to be a numpy.array, a vector of shape m * 1.
		theta: has to be a numpy.array, a 2 * 1 vector.
	Return:
		The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
		None if x, y, or theta is an empty numpy.ndarray.
		None if x, y and theta do not have compatible dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	if ((not isVector(x)) or (not isVector(y)) or y.shape[0] != x.shape[0] or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1):
		return None
	copyX = np.insert(x, 0, 1, axis=1)
	transpX = copyX.transpose()
	return np.matmul(transpX, (np.matmul(copyX, theta) - y)) / x.shape[0]
	

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))
	theta1 = np.array([2, 0.7]).reshape((-1, 1))
	theta2 = np.array([1, -0.4]).reshape((-1, 1))
	ret1 = simple_gradient(x, y, theta1)
	ret2 = simple_gradient(x, y, theta2)
	print(ret1)
	print(ret2)
	print("\n-------   Result expected   -------\n")
	print("## This is aproximative result")
	print(np.array([[-19.0342], [-586.6687]]))
	print(np.array([[-57.8682], [-2230.1229]]))