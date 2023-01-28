import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

def mse_(y, y_hat):
	"""
	Description:
		Calculate the MSE between the predicted output and the real output.
	Args:
		y: has to be a numpy.array, a vector of dimension m * 1.
		y_hat: has to be a numpy.array, a vector of dimension m * 1.
	Returns:
		mse: has to be a float.
		None if there is a matching dimension problem.
	Raises:
		This function should not raise any Exceptions.
	"""
	if (not check_matrix(y, -1, -1, 1) or not check_matrix(y_hat, y.shape[0], -1, 1)):
		return None
	return sum((y_hat - y) ** 2) / y.size

def rmse_(y, y_hat):
	"""
	Description:
		Calculate the RMSE between the predicted output and the real output.
	Args:
		y: has to be a numpy.array, a vector of dimension m * 1.
		y_hat: has to be a numpy.array, a vector of dimension m * 1.
	Returns:
		rmse: has to be a float.
		None if there is a matching dimension problem.
	Raises:
		This function should not raise any Exceptions.
	"""
	ret = mse_(y, y_hat)
	if (ret is None):
		return None
	return sqrt(ret)

def mae_(y, y_hat):
	"""
	Description:
		Calculate the MAE between the predicted output and the real output.
	Args:
		y: has to be a numpy.array, a vector of dimension m * 1.
		y_hat: has to be a numpy.array, a vector of dimension m * 1.
	Returns:
		mae: has to be a float.
		None if there is a matching dimension problem.
	Raises:
		This function should not raise any Exceptions.
	"""
	if (not check_matrix(y, -1, -1, 1) or not check_matrix(y_hat, y.shape[0], -1, 1)):
		return None
	return sum(abs(y_hat - y)) / y.size

def r2score_(y, y_hat):
	"""
	Description:
		Calculate the R2score between the predicted output and the output.
	Args:
		y: has to be a numpy.array, a vector of dimension m * 1.
		y_hat: has to be a numpy.array, a vector of dimension m * 1.
	Returns:
		r2score: has to be a float.
		None if there is a matching dimension problem.
	Raises:
		This function should not raise any Exceptions.
	"""
	if (not check_matrix(y, -1, -1, 1) or not check_matrix(y_hat, y.shape[0], -1, 1)):
		return None
	return 1 - (sum((y_hat - y) ** 2) / sum((y - y.mean()) ** 2))

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	x = np.array([0, 15, -9, 7, 12, 3, -21])
	y = np.array([2, 14, -13, 5, 12, 4, -19])
	ret1 = mse_(x,y)
	print(ret1)
	ret2 = rmse_(x,y)
	print(ret2)
	ret3 = mae_(x,y)
	print(ret3)
	ret4 = r2score_(x,y)
	print(ret4)
	print("\n-------   Result expected   -------\n")
	print("## With function of sklearn")
	ret5 = mean_squared_error(x,y)
	print(ret5)
	ret6 = sqrt(mean_squared_error(x,y))
	print(ret6)
	ret7 = mean_absolute_error(x,y)
	print(ret7)
	ret8 = r2_score(x,y)
	print(ret8)
	print("## Brute value")
	print(4.285714285714286)
	print(2.0701966780270626)
	print(1.7142857142857142)
	print(0.9681721733858745)