import numpy as np

#!########################################################################################################!#
#!##############################################  FUNCTION  ##############################################!#
#!########################################################################################################!#

def check_matrix(m, sizeX = -1, sizeY = -1, dim = 2):
	if (not isinstance(m, np.ndarray)):
		return False
	if (m.ndim != dim or m.size == 0):
		return False
	if (sizeX != -1 and m.shape[0] != sizeX):
		return False
	if (dim == 2 and sizeY != -1 and m.shape[1] != sizeY):
		return False
	return True

def iterative_l2(theta):
	"""Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
	Args:
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
	Returns:
		The L2 regularization as a float.
		None if theta in an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(theta, -1, 1)):
		return None
	ret = 0
	newTheta = np.copy(theta).reshape(-1)
	for i in range(1, newTheta.shape[0]):
		ret += newTheta[i] ** 2
	return ret


def l2(theta):
	"""Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
	Args:
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
	Returns:
		The L2 regularization as a float.
		None if theta in an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(theta, -1, 1)):
		return None
	newTheta = np.copy(theta).reshape(-1)
	newTheta[0] = 0
	return np.dot(newTheta, newTheta)

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	y = np.array([3,0.5,-6]).reshape((-1, 1))

	print(iterative_l2(x))
	print(l2(x))
	print(iterative_l2(y))
	print(l2(y))
	print("\n-------   Result expected   -------\n")
	print(911.0)
	print(911.0)
	print(36.25)
	print(36.25)