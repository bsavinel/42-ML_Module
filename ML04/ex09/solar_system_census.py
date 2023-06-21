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

Xtrain, Xeval, Ytrain, Yeval = data_spliter(data, result, 0.7, progSeed)
XtrainNorm = normalizer_multiline(Xtrain, data)
XevalNorm = normalizer_multiline(Xeval, data)
XtrainNorm = add_polynomial_features(XtrainNorm, 3)
XevalNorm = add_polynomial_features(XevalNorm, 3)

loss = [-1.,-1.,-1.,-1.]
lambdas = [0,0,0,0]

try:
	file = open('model.pickel', 'rb')
	data = pickle.load(file)
except:
	print("Error: model file opening failed")
	exit()
file.close()

for elem in data:
	if (loss[elem[0]] == -1. or elem[2] < loss[elem[0]]):
		lambdas[elem[0]] = elem[1]
		loss[elem[0]] = elem[2]

prediction = []
for i in range(4):
	YtmpTrain = np.where(Ytrain == i, 1, 0).reshape(-1, 1)
	YtmpEval = np.where(Yeval == i, 1, 0).reshape(-1, 1)
	myLR = MyLR(np.ones((10,1)), 0.1, 6000, penalty='l2', lambda_=lambdas[i])
	myLR.fit_(XtrainNorm, YtmpTrain)
	Yhat = myLR.predict_(XevalNorm)
	prediction.append(Yhat.reshape(-1))


resultComp = []
for i in range(len(prediction[0])):
	if (prediction[0][i] >= prediction[1][i] and prediction[0][i] >= prediction[2][i] and prediction[0][i] >= prediction[3][i]):
		resultComp.append(0)
	elif (prediction[1][i] >= prediction[0][i] and prediction[1][i] >= prediction[2][i] and prediction[1][i] >= prediction[3][i]):
		resultComp.append(1)
	elif (prediction[2][i] >= prediction[0][i] and prediction[2][i] >= prediction[1][i] and prediction[2][i] >= prediction[3][i]):
		resultComp.append(2)
	else:
		resultComp.append(3)

print("f1_score of the model is: ", MyLR.f1_score_(Yeval, np.array(resultComp)))


ax = plt.axes(projection='3d')
Yeval = Yeval.reshape(-1)
resultComp = np.array(resultComp).reshape(-1)
ax.plot(Yeval, Xeval[:,0], Xeval[:,1], label="True value", marker='.', linestyle='None')
ax.plot(resultComp, Xeval[:,0], Xeval[:,1], label="Prediction", marker='.', linestyle='None')
ax.set_title('Repartion in function of the weight and the height')
ax.set_xlabel("zipcode of the civilisation")
ax.set_ylabel('Weight')
ax.set_zlabel('Height')
ax.legend()
plt.show()

ax = plt.axes(projection='3d')
ax.plot(Yeval, Xeval[:,1], Xeval[:,2], label="True value", marker='.', linestyle='None')
ax.plot(resultComp, Xeval[:,1], Xeval[:,2], label="Prediction", marker='.', linestyle='None')
ax.set_title('Repartion in function of the height and the bone density')
ax.set_xlabel("zipcode of the civilisation")
ax.set_ylabel('Height')
ax.set_zlabel('Bone density')
ax.legend()
plt.show()

ax = plt.axes(projection='3d')
ax.plot(Yeval, Xeval[:,0], Xeval[:,2], label="True value", marker='.', linestyle='None')
ax.plot(resultComp, Xeval[:,0], Xeval[:,2], label="Prediction", marker='.', linestyle='None')
ax.set_title('Repartion in function of the weight and the bone density')
ax.set_xlabel("zipcode of the civilisation")
ax.set_ylabel('Weight')
ax.set_zlabel('Bone density')
ax.legend()
plt.show()