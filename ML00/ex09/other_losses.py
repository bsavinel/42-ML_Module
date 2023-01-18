import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#
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
	if ((not isinstance(y, np.ndarray)) or (not isinstance(y_hat, np.ndarray)) or y.ndim != 1 or y_hat.ndim != 1 or y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape):
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
	if ((not isinstance(y, np.ndarray)) or (not isinstance(y_hat, np.ndarray)) or y.ndim != 1 or y_hat.ndim != 1 or y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape):
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
	if ((not isinstance(y, np.ndarray)) or (not isinstance(y_hat, np.ndarray)) or y.ndim != 1 or y_hat.ndim != 1 or y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape):
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
	# print("## With function of sklearn")
	# ret5 = mean_squared_error(x,y)
	# print(ret5)
	# ret6 = sqrt(mean_squared_error(x,y))
	# print(ret6)
	# ret7 = mean_absolute_error(x,y)
	# print(ret7)
	# ret8 = r2_score(x,y)
	# print(ret8)
	print("## Brute value")
	print(4.285714285714286)
	print(2.0701966780270626)
	print(1.7142857142857142)
	print(0.9681721733858745)