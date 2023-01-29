import numpy as np

#!########################################################################################################!#
#!##############################################  FUNCTION  ##############################################!#
#!########################################################################################################!#


def check_matrix(m, sizeX, sizeY, dim = 2):
	if (not isinstance(m, np.ndarray)):
		return False
	if (m.ndim != dim or m.size == 0):
		return False
	if (sizeX != -1 and m.shape[0] != sizeX):
		return False
	if (dim == 2 and sizeY != -1 and m.shape[1] != sizeY):
		return False
	return True

def sigmoid_(x):
	"""
	Compute the sigmoid of a vector.
	Args:
		x: has to be a numpy.ndarray of shape (m, 1).
	Returns:
		The sigmoid value as a numpy.ndarray of shape (m, 1).
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, 1)):
		return None
	return 1 / (1 + np.exp(x * -1))	

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	x = np.array([[-4]])
	ret1 = sigmoid_(x)
	x = np.array([[2]])
	ret2 = sigmoid_(x)
	x = np.array([[-4], [2], [0]])
	ret3 = sigmoid_(x)
	print(ret1)
	print(ret2)
	print(ret3)
	print("\n-------   Result expected   -------\n")
	print(np.array([[0.01798620996209156]]))
	print(np.array([[0.8807970779778823]]))
	print(np.array([[0.01798620996209156], [0.8807970779778823], [0.5]]))