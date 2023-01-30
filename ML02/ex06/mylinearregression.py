import numpy as np
from math import sqrt

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

def isVector(x):
	if ((isinstance(x, np.ndarray)) and (x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1))):
		return True
	return False

def gradient(x, y, theta):
	if (not check_matrix(x, -1, -1) or not check_matrix(y, x.shape[0], 1) or not check_matrix(theta, x.shape[1] + 1, 1)):
		return None
	xCopy = np.insert(x,0, 1, axis = 1)
	return (np.matmul(xCopy.transpose(), np.matmul(xCopy, theta) - y)) / x.shape[0]

#!##########################################################################################!#
#!########################################  Classe  ########################################!#
#!##########################################################################################!#

class MyLinearRegression():

#!---------------- Constructeur ----------------!#

	def __init__(self, theta, alpha=0.001, max_iter=1000):
		if ((not isinstance(alpha, float)) or (not isinstance(max_iter, int)) or max_iter < 0 or theta.shape[1] != 1):
			raise ValueError 
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta
	
#!---------------- Méthodes ----------------!#

	def fit_(self, x, y):
		if (not check_matrix(x, -1, self.theta.shape[0] - 1) or not check_matrix(y, x.shape[0], 1)):
			return None
		for i in range(self.max_iter):
			self.theta = self.theta - (self.alpha * gradient(x, y, self.theta))

	def predict_(self, x):
		if (not check_matrix(x, -1, -1) or not check_matrix(self.theta, x.shape[1] + 1, 1)):
			return None
		newX = np.insert(x, 0, 1, axis = 1) #add 1 to the first collum to have the first theta value as constant
		return (newX @ self.theta).astype(float)

	def loss_elem_(self, y, y_hat):
		if ((not isVector(y)) or (not isVector(y_hat))):
			return None
		copyY = y.copy()
		copyYHat = y_hat.copy()
		copyY = copyY.reshape(-1, 1)
		copyYHat = copyYHat.reshape(-1, 1)
		if (not check_matrix(copyY, -1, 1) or not check_matrix(copyYHat, copyYHat.shape[0], 1)):
			return None
		return np.array([(copyY[i] - copyYHat[i]) ** 2 for i in range(copyY.shape[0])])

	def loss_(self, y, y_hat):
		ret = self.loss_elem_(y, y_hat)
		if (ret is None):
			return None
		return np.sum(ret) / (2 * y.shape[0])
	
#! ---------------- Static function ---------------- !#

	@staticmethod
	def mse_(y, y_hat):
		if (check_matrix(y, -1, 1)):
			y = y.reshape(-1)
		if (check_matrix(y_hat, -1, 1)):
			y_hat = y_hat.reshape(-1)
		if (not check_matrix(y, -1, -1, 1) or not check_matrix(y_hat, y.shape[0], -1, 1)):
			return None
		return sum((y_hat - y) ** 2) / y.size

	@staticmethod
	def rmse_(y, y_hat):
		ret = MyLinearRegression.mse_(y, y_hat)
		if (ret is None):
			return None
		return sqrt(ret)

	@staticmethod
	def mae_(y, y_hat):
		if (not check_matrix(y, -1, -1, 1) or not check_matrix(y_hat, y.shape[0], -1, 1)):
			return None
		return sum(abs(y_hat - y)) / y.size

	@staticmethod
	def r2score_(y, y_hat):
		if (not check_matrix(y, -1, -1, 1) or not check_matrix(y_hat, y.shape[0], -1, 1)):
			return None
		return 1 - (sum((y_hat - y) ** 2) / sum((y - y.mean()) ** 2))