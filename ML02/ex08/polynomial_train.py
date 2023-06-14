import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR

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

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	data = pd.read_csv("are_blue_pills_magics.csv")
	Xpill = np.array(data["Micrograms"]).reshape(-1,1)
	Yscore = np.array(data["Score"]).reshape(-1,1)
	theta1 = np.array([[42.4852],[-1.0316]]).reshape(-1,1)
	theta2 = np.array([[58.0258],[-2.6528],[0.0285]]).reshape(-1,1)
	theta3 = np.array([[71.2728],[-4.2336],[0.0922],[-0.0007]]).reshape(-1,1)
	theta4 = np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
	theta5 = np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1)
	theta6 = np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]).reshape(-1,1)
	
	linearModel1 = MyLR(theta1, alpha=2.5e-5)
	linearModel2 = MyLR(theta2, alpha=2.5e-5)
	linearModel3 = MyLR(theta3, alpha=2.5e-5, max_iter=6000)
	linearModel4 = MyLR(theta4, alpha=1.5e-6, max_iter=3500)
	linearModel5 = MyLR(theta5, alpha=4e-8, max_iter=3000)
	linearModel6 = MyLR(theta6, alpha=1e-9, max_iter=400)
	linearModel1.fit_(Xpill, Yscore)
	linearModel2.fit_(add_polynomial_features(Xpill, 2), Yscore)
	linearModel3.fit_(add_polynomial_features(Xpill, 3), Yscore)
	linearModel4.fit_(add_polynomial_features(Xpill, 4), Yscore)
	linearModel5.fit_(add_polynomial_features(Xpill, 5), Yscore)
	linearModel6.fit_(add_polynomial_features(Xpill, 6), Yscore)

	predict1 = linearModel1.predict_(Xpill)
	predict2 = linearModel2.predict_(add_polynomial_features(Xpill, 2))
	predict3 = linearModel3.predict_(add_polynomial_features(Xpill, 3))
	predict4 = linearModel4.predict_(add_polynomial_features(Xpill, 4))
	predict5 = linearModel5.predict_(add_polynomial_features(Xpill, 5))
	predict6 = linearModel6.predict_(add_polynomial_features(Xpill, 6))

	print(MyLR.mse_(Yscore, predict1))
	print(MyLR.mse_(Yscore, predict2))
	print(MyLR.mse_(Yscore, predict3))
	print(MyLR.mse_(Yscore, predict4))
	print(MyLR.mse_(Yscore, predict5))
	print(MyLR.mse_(Yscore, predict6))
	
	plt.plot(Xpill, Yscore, 'bo')
	plt.plot(Xpill, predict1, 'r.')
	plt.plot(Xpill, predict2, 'y.')
	plt.plot(Xpill, predict3, 'g.')
	plt.plot(Xpill, predict4, 'c.')
	plt.plot(Xpill, predict5, 'm.')
	plt.plot(Xpill, predict6, 'k.')
	plt.show()