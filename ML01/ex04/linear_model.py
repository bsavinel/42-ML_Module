import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR

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

def add_intercept(x):
	"""Adds a column of 1's to the non-empty numpy.array x.
	Args:
		x: has to be a numpy.array of dimension m * n.
	Returns:
		X, a numpy.array of dimension m * (n + 1).
		None if x is not a numpy.array.
		None if x is an empty numpy.array.
	Raises:
		This function should not raise any Exception.
	"""
	if ((not isinstance(x, np.ndarray)) or x.size == 0):
		return None
	tmp = x.copy()
	if (tmp.ndim == 1):
		tmp.resize((tmp.shape[0], 1))
	return np.insert(tmp, 0, 1, axis=1)

def predict_(x, theta):
	if (not check_matrix(x, -1, 1) or not check_matrix(theta, 2, 1)):
		return None
	return np.dot(add_intercept(x), theta)


def plot(x, y, theta, title = "Linear Regression", xlabel = "x", ylabel = "y"):
	if ((not isinstance(x, np.ndarray)) or (not isinstance(y, np.ndarray)) or (not isinstance(theta, np.ndarray)) or x.size == 0 or y.size == 0 or theta.size == 0 or x.ndim != 1 or y.ndim != 1 or theta.ndim != 2 or x.shape[0] != y.shape[0] or theta.shape[0] != 2 or theta.shape[1] != 1):
		return
	if ((not isinstance(title, str)) or (not isinstance(xlabel, str)) or (not isinstance(ylabel, str))):
		return
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(x, predict_(x.reshape((-1,1)), theta), 'sy--')
	plt.plot(x, y, 'bo')
	plt.grid()
	plt.xlabel("Quantity of blue pill (in micrograms)")
	plt.ylabel("Spade driving score")
	plt.legend(['$S_{predier}(pills)$', '$S_{true}(pills)$'])
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

	linear_model3 = MyLR(np.array([[0.0], [0.0]]))
	valTheta0 = np.linspace(80, 96, 6)
	valTheta1 = np.linspace(-14, -3, 100)
	evol = np.linspace(-14, -3, 100)
	fig, axe = plt.subplots()
	for i in valTheta0:
		val = np.array([])
		linear_model3.thetas[0] = i
		for j in valTheta1:
			linear_model3.thetas[1] = j
			val = np.append(val, linear_model3.loss_(Yscore, linear_model3.predict_(Xpill)))
		plt.plot(evol, val, '-')
	axe.set_ylim([10, 150])
	plt.grid()
	plt.title("Evolution of the loss for different values of $θ_0$")
	plt.ylabel('cost function $J(θ_0,θ_1)$')
	plt.xlabel('$θ_1$')
	plt.legend(['$J((θ_0=c_0,θ_1)$',
				'$J((θ_0=c_1,θ_1)$',
				'$J((θ_0=c_2,θ_1)$',
				'$J((θ_0=c_3,θ_1)$',
				'$J((θ_0=c_4,θ_1)$',
				'$J((θ_0=c_5,θ_1)$',])
	plt.show()



	print("\n-------   Result with my linear regression   -------\n")
	print(MyLR.mse_(Yscore, Y_model1))
	print(MyLR.mse_(Yscore, Y_model2))
	print("\n-------   Result with sklearn   -------\n")
	print(mean_squared_error(Yscore, Y_model1))
	print(mean_squared_error(Yscore, Y_model2))
	print("\n-------   Result expected   -------\n")
	print(57.60304285714282)
	print(232.16344285714285)