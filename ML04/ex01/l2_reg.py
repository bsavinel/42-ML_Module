import numpy as np

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
	newTheta = np.copy(theta).reshape(-1)
	newTheta[0] = 0
	return np.dot(newTheta, newTheta)

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