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

def format_prediction(a,b,c,d):
	resultComp = []
	for i in range(len(a)):
		if (a[i] >= b[i] and a[i] >= c[i] and a[i] >= d[i]):
			resultComp.append(0)
		elif (b[i] >= a[i] and b[i] >= c[i] and b[i] >= d[i]):
			resultComp.append(1)
		elif (c[i] >= a[i] and c[i] >= b[i] and c[i] >= d[i]):
			resultComp.append(2)
		else:
			resultComp.append(3)
	return np.array(resultComp).reshape(-1)

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


try:
	file = open('model.pickel', 'rb')
	data = pickle.load(file)
except:
	print("Error: model file opening failed")
	exit()
file.close()

predTab = []
for i in range(len(data)):
	predTab.append([[],[],[],[]])

for i in range(len(data)):
	for j in range(4):
		model = MyLR(data[i][j][1], lambda_=data[i][j][0])
		pred = model.predict_(XtrainNorm)
		predTab[i][j] = pred


F1ScoreTab = []
bestMean = 0
bestLambda = 0
for i in range(len(predTab)):
	OriginPred = format_prediction(predTab[i][0], predTab[i][1], predTab[i][2], predTab[i][3])
	f1Tab = []
	for j in range(4):
		f1Tab.append(MyLR.f1_score_(Ytrain, OriginPred, pos_label=j))
	f1Tab.append(np.mean(f1Tab))
	print("Lambda = ", data[i][0][0])
	print("\tF1 score for origine 0: ", f1Tab[0])
	print("\tF1 score for origine 1: ", f1Tab[1])
	print("\tF1 score for origine 2: ", f1Tab[2])
	print("\tF1 score for origine 3: ", f1Tab[3])
	print("\tMean F1 score: ", f1Tab[4])
	F1ScoreTab.append(f1Tab)
	if (f1Tab[4] >= bestMean):
		bestMean = f1Tab[4]
		bestLambda = data[i][0][0]
print("\nBest lambda: ", bestLambda)

F1ScoreTab =np.array(F1ScoreTab)

lambdas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.bar(lambdas, F1ScoreTab[:,0], width=0.1, color='r', label='Origine 0')
plt.title("F1 score for origine 0")
plt.xlabel("Lambda")
plt.ylabel("F1 score")
plt.legend()
plt.show()
plt.bar(lambdas, F1ScoreTab[:,1], width=0.1, color='g', label='Origine 1')
plt.title("F1 score for origine 1")
plt.xlabel("Lambda")
plt.ylabel("F1 score")
plt.legend()
plt.show()
plt.bar(lambdas, F1ScoreTab[:,2], width=0.1, color='b', label='Origine 2')
plt.title("F1 score for origine 2")
plt.xlabel("Lambda")
plt.ylabel("F1 score")
plt.legend()
plt.show()
plt.bar(lambdas, F1ScoreTab[:,3], width=0.1, color='y', label='Origine 3')
plt.title("F1 score for origine 3")
plt.xlabel("Lambda")
plt.ylabel("F1 score")
plt.legend()
plt.show()
plt.bar(lambdas, F1ScoreTab[:,4], width=0.1, color='y', label='Mean')
plt.title("F1 score mean")
plt.xlabel("Lambda")
plt.ylabel("F1 score")
plt.legend()
plt.show()


prediction = []
for i in range(4):
	YtmpTrain = np.where(Ytrain == i, 1, 0).reshape(-1, 1)
	YtmpEval = np.where(Yeval == i, 1, 0).reshape(-1, 1)
	myLR = MyLR(np.ones((10,1)), 0.1, 30000, penality='l2', lambda_=bestLambda)
	myLR.fit_(XtrainNorm, YtmpTrain)
	prediction.append(myLR.predict_(XevalNorm).reshape(-1))


OriginPred = format_prediction(prediction[0], prediction[1], prediction[2], prediction[3])
for j in range(4):
	f1Tab.append(MyLR.f1_score_(Yeval, OriginPred, pos_label=j))
print("\nBest model:")
print("\tF1 score for origine 0: ", f1Tab[0])
print("\tF1 score for origine 1: ", f1Tab[1])
print("\tF1 score for origine 2: ", f1Tab[2])
print("\tF1 score for origine 3: ", f1Tab[3])
print("\tMean F1 score: ", np.mean(f1Tab))


ax = plt.axes(projection='3d')
Yeval = Yeval.reshape(-1)
OriginPred = np.array(OriginPred).reshape(-1)
ax.plot(Yeval, Xeval[:,0], Xeval[:,1], label="True value", marker='.', linestyle='None')
ax.plot(OriginPred, Xeval[:,0], Xeval[:,1], label="Prediction", marker='.', linestyle='None')
ax.set_title('Repartion in function of the weight and the height')
ax.set_xlabel("zipcode of the civilisation")
ax.set_ylabel('Weight')
ax.set_zlabel('Height')
ax.legend()
plt.show()

ax = plt.axes(projection='3d')
ax.plot(Yeval, Xeval[:,1], Xeval[:,2], label="True value", marker='.', linestyle='None')
ax.plot(OriginPred, Xeval[:,1], Xeval[:,2], label="Prediction", marker='.', linestyle='None')
ax.set_title('Repartion in function of the height and the bone density')
ax.set_xlabel("zipcode of the civilisation")
ax.set_ylabel('Height')
ax.set_zlabel('Bone density')
ax.legend()
plt.show()

ax = plt.axes(projection='3d')
ax.plot(Yeval, Xeval[:,0], Xeval[:,2], label="True value", marker='.', linestyle='None')
ax.plot(OriginPred, Xeval[:,0], Xeval[:,2], label="Prediction", marker='.', linestyle='None')
ax.set_title('Repartion in function of the weight and the bone density')
ax.set_xlabel("zipcode of the civilisation")
ax.set_ylabel('Weight')
ax.set_zlabel('Bone density')
ax.legend()
plt.show()