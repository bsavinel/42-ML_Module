import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR


def predict_(x, theta):

	if ((not isinstance(x, np.ndarray)) or (not isinstance(theta, np.ndarray))  or x.size == 0 or theta.size != 2 or x.ndim != 1 or theta.ndim != 2):
		return None
	tmp = np.array([float(theta[0] + theta[1] * x[i]) for i in range(x.shape[0])])
	return tmp

def plot(x, y, theta, title = "Linear Regression", xlabel = "x", ylabel = "y"):
	if ((not isinstance(x, np.ndarray)) or (not isinstance(y, np.ndarray)) or (not isinstance(theta, np.ndarray)) or x.size == 0 or y.size == 0 or theta.size == 0 or x.ndim != 1 or y.ndim != 1 or theta.ndim != 2 or x.shape[0] != y.shape[0] or theta.shape[0] != 2 or theta.shape[1] != 1):
		return
	if ((not isinstance(title, str)) or (not isinstance(xlabel, str)) or (not isinstance(ylabel, str))):
		return
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(x, y, 'bo')
	plt.plot(x, predict_(x, theta), 'sy--')
	plt.grid()
	plt.show()

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	data = pd.read_csv("are_blue_pills_magics.csv")
	Xpill = np.array(data["Micrograms"]).reshape(-1,1)
	Yscore = np.array(data["Score"]).reshape(-1,1)

	linear_model1 = MyLR(np.array([[89.0], [-8]]))
	linear_model2 = MyLR(np.array([[89.0], [-6]]))

	Y_model1 = linear_model1.predict_(Xpill)
	Y_model2 = linear_model2.predict_(Xpill)
	linear_model1.predict_(Xpill)
	Yscore = Yscore.reshape(-1)
	Y_model1 = Y_model1.reshape(-1)
	Y_model2 = Y_model2.reshape(-1)
	plot(Xpill.reshape(-1), Yscore.reshape(-1), linear_model1.thetas, "Linear Regression before fit", "Micrograms", "Score")
	plot(Xpill.reshape(-1), Yscore.reshape(-1), linear_model2.thetas, "Linear Regression before fit", "Micrograms", "Score")
	linear_model1.fit_(Xpill.reshape(-1,1), Yscore.reshape(-1,1))
	linear_model2.fit_(Xpill.reshape(-1,1), Yscore.reshape(-1,1))
	plot(Xpill.reshape(-1), Yscore.reshape(-1), linear_model1.thetas, "Linear Regression after fit", "Micrograms", "Score")
	plot(Xpill.reshape(-1), Yscore.reshape(-1), linear_model2.thetas, "Linear Regression after fit", "Micrograms", "Score")

	print("\n-------   Result with my linear regression   -------\n")
	print(MyLR.mse_(Yscore, Y_model1))
	print(MyLR.mse_(Yscore, Y_model2))
	print("\n-------   Result with sklearn   -------\n")
	print(mean_squared_error(Yscore, Y_model1))
	print(mean_squared_error(Yscore, Y_model2))
	print("\n-------   Result expected   -------\n")
	print(57.60304285714282)
	print(232.16344285714285)