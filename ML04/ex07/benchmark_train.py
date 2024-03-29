import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ridge import MyRidge, check_matrix
import pickle

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

def plot_evolution(los_evolution, pointType):
	"""Plot the evolution of the loss function."""
	if (not isinstance(los_evolution, list)):
		return None
	plt.plot(los_evolution, pointType)
	

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

def add_polynomial_features(x, power):
	"""Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power give
	Args:
		x: has to be an numpy.ndarray, a matrix of shape m * n.
		power: has to be an int, the power up to which the columns of matrix x are going to be raised.
	Returns:
		The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature va
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, -1) or not isinstance(power, int)):
		return None
	ret = np.empty((x.shape[0], 0), dtype=int)
	for i in range(1, power + 1):
		ret = np.concatenate((ret, x ** i), axis=1)
	return ret

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	data = pd.read_csv("space_avocado.csv")
	X = np.array(data[["weight", "prod_distance", "time_delivery"]]).reshape(-1,3)
	Y = np.array(data["target"]).reshape(-1,1)
	Xnorm = normalizer(X, X)
	Ynorm = normalizer(Y, Y).reshape(-1,1)
	Xtrain, Xeval, Ytrain, Yeval = data_spliter(Xnorm, Ynorm, 0.8)

	color = ["b", "r", "g", "y", "c", "m", "k", "w"]

	key = "{} polinomial features with lambda {}"
	labelKey = "lambda = {}"
	titleKey = "Evolution of the loss function during the fit for {} polinomial features"
	model_loss = {}
	lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

	tab = []

	for i in range(0,4):
		tmpXtrain = add_polynomial_features(Xtrain, i + 1)
		tmpXeval =  add_polynomial_features(Xeval, i + 1)
		theta =  np.ones((Xtrain.shape[1] * (i + 1) + 1, 1))
		for j in lambdas:
			linearModel = MyRidge(theta, alpha=0.001, max_iter=3000, lambda_=j)
			linearModel.fit_(tmpXtrain, Ytrain)
			# print(linearModel.thetas)
			predict = linearModel.predict_(tmpXeval)
			loss = linearModel.mse_(Yeval.reshape(-1), predict.reshape(-1))
			loss = linearModel.loss_(Yeval, predict)
			print("loss of", key.format(i + 1, j),":", loss)
			plt.plot(linearModel.loss_evolution, color[int(j * 2.5)], label=labelKey.format(j))
			model_loss[key.format(i + 1, j)] = loss
			tab.append([i + 1, j, linearModel.thetas])
		plt.title(titleKey.format(i + 1))
		plt.xlabel("Iteration")
		plt.ylabel("Loss")
		plt.legend()
		plt.show()


	file = open('model.pickel', 'wb')
	pickle.dump(tab, file)
	file.close()


	print ("\nThe best is with :", min(model_loss, key=model_loss.get))
