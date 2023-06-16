import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from my_logistic_regression import check_matrix, MyLogisticRegression as MyLR
import sys
import random

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

#!#################################################################################################
#!#####################################   Programe   ##############################################
#!#################################################################################################

zipcode = random.randint(0, 3)
for (i, arg) in enumerate(sys.argv):
	if (arg.startswith('-zipcode=') or arg.startswith('zipcode=–') and len(arg) == 10):
		try:
			zipcode = int(arg[9])
		except:
			print("Error: zipcode must be 0, 1, 2 or 3")
		break

if (zipcode == 0):
	planet = "The flying cities of Venus"
elif (zipcode == 1):
	planet = "United Nations of Earth"
elif (zipcode == 2):
	planet = "Mars Republic"
elif (zipcode == 3):
	planet = "The Asteroids' Belt colonies"

print("The zipcode", zipcode, "is for the citizen of", planet)

progSeed = int(datetime.now().timestamp())
data = pd.read_csv("solar_system_census.csv")
result = pd.read_csv("solar_system_census_planets.csv")
data = np.array(data[["weight","height","bone_density"]])
result = np.array(result["Origin"])
result = np.where(result == zipcode, 1, 0).reshape(-1, 1)
Xtrain, Xeval, Ytrain, Yeval = data_spliter(data, result, 0.7, progSeed)
XtrainNorm = normalizer_multiline(Xtrain, data)
XevalNorm = normalizer_multiline(Xeval, data)


myLR = MyLR(np.ones((4,1)), 0.1, 150000)
myLR.fit_(XtrainNorm, Ytrain)
Yhat = myLR.predict_(XevalNorm)
YhatComp = np.where(Yhat >= 0.5, 1, 0)


count = 0
for i in range(Yeval.shape[0]):
	if (Yeval[i] == YhatComp[i]):
		count += 1
print("Accuracy:", count / Yeval.shape[0])


ax = plt.axes(projection='3d')
Yeval = Yeval.reshape(-1)
Yhat = Yhat.reshape(-1)
ax.plot(Yeval, Xeval[:,0], Xeval[:,1], label="True value", marker='.', linestyle='None')
ax.plot(Yhat, Xeval[:,0], Xeval[:,1], label="Prediction", marker='.', linestyle='None')
ax.set_title('Repartion in function of the weight and the height')
ax.set_xlabel('Is a citizen of {}'.format(planet))
ax.set_ylabel('Weight')
ax.set_zlabel('Height')
ax.legend()
plt.show()

ax = plt.axes(projection='3d')
ax.plot(Yeval, Xeval[:,1], Xeval[:,2], label="True value", marker='.', linestyle='None')
ax.plot(Yhat, Xeval[:,1], Xeval[:,2], label="Prediction", marker='.', linestyle='None')
ax.set_title('Repartion in function of the height and the bone density')
ax.set_xlabel('Is a citizen of {}'.format(planet))
ax.set_ylabel('Height')
ax.set_zlabel('Bone density')
ax.legend()
plt.show()

ax = plt.axes(projection='3d')
ax.plot(Yeval, Xeval[:,0], Xeval[:,2], label="True value", marker='.', linestyle='None')
ax.plot(Yhat, Xeval[:,0], Xeval[:,2], label="Prediction", marker='.', linestyle='None')
ax.set_title('Repartion in function of the weight and the bone density')
ax.set_xlabel('Is a citizen of {}'.format(planet))
ax.set_ylabel('Weight')
ax.set_zlabel('Bone density')
ax.legend()
plt.show()
