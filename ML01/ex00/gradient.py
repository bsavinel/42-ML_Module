import numpy as np

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
	if (not check_matrix(x, -1, 1) or not check_matrix(y, x.shape[0], 1) or not check_matrix(theta, 2, 1)):
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