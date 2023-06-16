import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR, check_matrix

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

def add_polynomial_features(x, power):
	"""Add polynomial features to vector x by raising its values up to the power given in argument.
	Args:
		x: has to be an numpy.array, a vector of dimension m * 1.
		power: has to be an int, the power up to which the components of vector x are going to be raised.
	Return:
		The matrix of polynomial features as a numpy.array, of dimension m * n,
		containing the polynomial feature values for all training examples.
		None if x is an empty numpy.array.
		None if x or power is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, 1) or not isinstance(power, int)):
		return None
	newx = np.repeat(x, power, axis=1)
	puiss = np.arange(1, power + 1)
	return np.power(newx, puiss)

def unison_shuffled_copies(a, b):
	"""Shuffle two arrays in the same way."""
	if (len(a) != len(b) or len(a) == 0):
		return None
	p = np.random.permutation(len(a))
	return a[p], b[p]

def data_spliter(x, y, proportion):
	"""Shuffles and splits the dataset (given by x and y) into a training and a test set,
	while respecting the given proportion of examples to be kept in the training set.
	Args:
		x: has to be an numpy.array, a matrix of dimension m * n.
		y: has to be an numpy.array, a vector of dimension m * 1.
		proportion: has to be a float, the proportion of the dataset that will be assigned to the
		training set.
	Return:
		(x_train, x_test, y_train, y_test) as a tuple of numpy.array
		None if x or y is an empty numpy.array.
		None if x and y do not share compatible dimensions.
		None if x, y or proportion is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, -1) or not check_matrix(y, x.shape[0], 1) or proportion > 1 or proportion < 0):
		return None
	index_prop = int(x.shape[0] * proportion)
	copyX, copyY = unison_shuffled_copies(x, y)
	return (copyX[:index_prop], copyX[index_prop:], copyY[:index_prop], copyY[index_prop:])

def normalizer(x, list1):
	if (not isinstance(x, np.ndarray) or x.size == 0):
		return None
	if (not check_matrix(x, -1, -1) and not check_matrix(x, -1, -1, 1)):
		return None
	for i in range(x.shape[1]):
		min = np.min(list1[:,i])
		max = np.max(list1[:,i])
		if (max == min):
			raise ValueError("Normalizer: max and min of the array are equal")
		tmp = ((x[:,i] - min) / (max - min)).reshape(-1,1)
		if (i == 0):
			ret = tmp
		else:
			ret = np.concatenate((ret, tmp), axis=1)
	return (ret)

def denormalizer(x, list1):
	if (not isinstance(x, np.ndarray) or x.size == 0):
		return None
	if (not check_matrix(x, -1, -1) and not check_matrix(x, -1, -1, 1)):
		return None
	for i in range(x.shape[1]):
		min = np.min(list1[:,i])
		max = np.max(list1[:,i])
		if (max == min):
			raise ValueError("Normalizer: max and min of the array are equal")
		tmp = (x[:,i] * (max - min) + min).reshape(-1,1)
		if (i == 0):
			ret = tmp
		else:
			ret = np.concatenate((ret, tmp), axis=1)
	return (ret)

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	data = pd.read_csv("space_avocado.csv")
	X = np.array(data[["weight", "prod_distance", "time_delivery"]]).reshape(-1,3)
	Y = np.array(data["target"]).reshape(-1,1)
	
	Xnorm = normalizer(X, X)
	# X = np.array([[15, 90, 36], [84, 55, 66], [87, 84, 29]])
	# Xnorm =normalizer( X,X )
	# print(Xnorm)
	# print(denormalizer(Xnorm, X))
	Ynorm = normalizer(Y, Y).reshape(-1,1)

	Xtrain, Xeval, Ytrain, Yeval = data_spliter(Xnorm, Ynorm, 0.8)
	theta = np.array([[0],[20],[0],[-10]]).reshape(-1,1)
	
	linearModel = MyLR(theta, alpha=0.1, max_iter=100000)
	linearModel.fit_(Xtrain, Ytrain)

	predict = linearModel.predict_(Xeval)
	print(linearModel.theta)

	print(MyLR.r2score_(Yeval.reshape(-1), predict.reshape(-1)))
	TrueXEval = denormalizer(Xeval, X)
	TrueYEval = denormalizer(Yeval, Y)
	TruePredict = denormalizer(predict, Y)
	plt.plot(TrueXEval[:,0], TrueYEval, 'bo', label="Price")
	plt.plot(TrueXEval[:,0], TruePredict, 'g.', label="Predicted price")
	plt.ylabel("Weight")
	plt.xlabel("Price")
	plt.legend()
	plt.show()

	plt.plot(TrueXEval[:,1], TrueYEval, 'bo', label="Price")
	plt.plot(TrueXEval[:,1], TruePredict, 'g.', label="Predicted price")
	plt.ylabel("Distance")
	plt.xlabel("Price")
	plt.legend()
	plt.show()

	plt.plot(TrueXEval[:,2], TrueYEval, 'bo', label="Price")
	plt.plot(TrueXEval[:,2], TruePredict, 'g.', label="Predicted price")
	plt.ylabel("Time")
	plt.xlabel("Price")
	plt.legend()
	plt.show()