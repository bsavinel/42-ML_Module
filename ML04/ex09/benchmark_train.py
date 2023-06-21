import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from my_logistic_regression import check_matrix, MyLogisticRegression as MyLR
import pickle

def unison_shuffled_copies(a, b, seed):
	if (len(a) != len(b) or len(a) == 0):
		return None
	np.random.seed(seed)
	p = np.random.permutation(len(a))
	return a[p], b[p]

def data_spliter(x, y, proportion, seed):
	if (not check_matrix(x, -1, -1) or not check_matrix(y, x.shape[0], 1) or proportion > 1 or proportion < 0 or not isinstance(seed, int)):
		return None
	index_prop = int(x.shape[0] * proportion)
	copyX, copyY = unison_shuffled_copies(x, y, seed)
	return (copyX[:index_prop], copyX[index_prop:], copyY[:index_prop], copyY[index_prop:])

def normalizer(x, list1):
	if (not check_matrix(x, -1, 1) or not check_matrix(list1, -1, 1)):
		print("error",x.shape)

		return None
	Xcopy = x.reshape(-1)
	min = np.min(list1)
	max = np.max(list1)
	if (max == min):
		raise ValueError("Normalizer: max and min of the array are equal")
	return (Xcopy - min) / (max - min)

def denormalizer(x, list1):
	if (not check_matrix(x, -1, 1) or not check_matrix(list1, -1, 1)):
		return None
	Xcopy = x.reshape(-1)
	min = np.min(list1)
	max = np.max(list1)
	return Xcopy * (max - min) + min

def normalizer_multiline(x, list):
	if (not check_matrix(x, -1, -1) or not check_matrix(list, -1, x.shape[1])):
		return None
	Xcopy = x.copy()
	for i in range(x.shape[1]):
		Xcopy[:, i] = normalizer(Xcopy[:, i].reshape(-1,1), list[:, i].reshape(-1,1))
	return Xcopy

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

#!#################################################################################################
#!#####################################   Programe   ##############################################
#!#################################################################################################

progSeed = int(datetime.now().timestamp())
data = pd.read_csv("solar_system_census.csv")
result = pd.read_csv("solar_system_census_planets.csv")
data = np.array(data[["weight","height","bone_density"]])
result = np.array(result["Origin"]).reshape(-1, 1)

prediction = [[],[],[],[],[]]
Xtrain, Xeval, Ytrain, Yeval = data_spliter(data, result, 0.7, progSeed)
XtrainNorm = normalizer_multiline(Xtrain, data)
XevalNorm = normalizer_multiline(Xeval, data)

lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
color = ["b", "r", "g", "y", "c", "m", "k", "w"]

XtrainNorm = add_polynomial_features(XtrainNorm, 3)
XevalNorm = add_polynomial_features(XevalNorm, 3)

tab = []

for i in range(4):
	YtmpTrain = np.where(Ytrain == i, 1, 0).reshape(-1, 1)
	YtmpEval = np.where(Yeval == i, 1, 0).reshape(-1, 1)
	for j in lambdas:
		myLR = MyLR(np.ones((10,1)), 0.1, 6000, penalty='l2', lambda_=j)
		myLR.fit_(XtrainNorm, YtmpTrain)
		Yhat = myLR.predict_(XevalNorm)
		prediction[int(j * 2.5)].append(Yhat.reshape(-1))
		plt.plot(myLR.loss_evolution, color[int(j * 2.5)], label="lambda: {}".format(j))
		print("Model for origine number",i,"with lamba", j, ":")
		print("/tLoss:", myLR.loss_(YtmpEval, Yhat), "\tF1_score:")
		tab.append([i, j, myLR.loss_(YtmpEval, Yhat)])
	print()
	plt.xlabel("Iteration")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()


file = open('model.pickel', 'wb')
pickle.dump(tab, file)
file.close()

