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
	if (not check_matrix(x, -1, -1, 1) or not check_matrix(theta, 2, 1)):
		return None
	tmp = np.array([float(theta[0] + theta[1] * x[i]) for i in range(x.shape[0])])
	return tmp.reshape((tmp.shape[0], 1))
	
#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	x = np.arange(1,6)

	theta1 = np.array([[5], [0]])
	theta2 = np.array([[0], [1]])
	theta3 = np.array([[5], [3]])
	theta4 = np.array([[-3], [1]])
	ret1 = predict_(x, theta1)
	ret2 = predict_(x, theta2)
	ret3 = predict_(x, theta3) # c'est X dans le sujet mais sans doute faute de frappe
	ret4 = predict_(x, theta4)
	print(ret1)
	print(ret2)
	print(ret3)
	print(ret4)
	print("\n-------   Result expected   -------\n")
	print(np.array([[5.], [5.], [5.], [5.], [5.]]))
	print(np.array([[1.], [2.], [3.], [4.], [5.]]))
	print(np.array([[ 8.], [11.], [14.], [17.], [20.]]))
	print(np.array([[-2.], [-1.], [ 0.], [ 1.], [ 2.]]))