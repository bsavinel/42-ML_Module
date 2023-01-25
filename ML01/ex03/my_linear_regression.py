import numpy as np

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

def isVector(x):
	if ((isinstance(x, np.ndarray)) and (x.ndim == 2 and x.shape[1] == 1)):
		return True
	return False

def simple_gradient(x, y, theta):
	if ((not isVector(x)) or (not isVector(y)) or y.shape[0] != x.shape[0] or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1):
		return None
	nabla0 = float((sum(((theta[1] * x) + theta[0]) - y)) / x.shape[0])
	nabla1 = float((sum((((theta[1] * x) + theta[0]) - y) * x)) / x.shape[0])
	return np.array([[nabla0], [nabla1]])

#!##########################################################################################!#
#!########################################  Classe  ########################################!#
#!##########################################################################################!#

class MyLinearRegression():

#!---------------- Constructeur ----------------!#

	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		if ((not isinstance(alpha, float)) or (not isinstance(max_iter, int)) or max_iter < 0 or thetas.ndim != 2 or thetas.shape[0] != 2 or thetas.shape[1] != 1):
			raise ValueError 
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas
	
#!---------------- Méthodes ----------------!#

	def fit_(self, x, y):
		if ((not isVector(x)) or (not isVector(y)) or y.shape[0] != x.shape[0]):
			return None
		for i in range(self.max_iter):
			self.thetas = self.thetas - (self.alpha * simple_gradient(x, y, self.thetas))

	def predict_(self, x):
		if ((not isinstance(x, np.ndarray)) or x.size == 0 or x.shape[1] != 1):
			return None
		copyX = x.reshape(-1)
		tmp = np.array([float(self.thetas[0] + self.thetas[1] * copyX[i]) for i in range(copyX.shape[0])])
		return tmp.reshape((tmp.shape[0], 1))

	def loss_elem_(self, y, y_hat):
		if ((not isVector(y)) or (not isVector(y_hat))):
			return None
		copyY = y.copy()
		copyYHat = y_hat.copy()
		copyY.reshape(-1, 1)
		copyYHat.reshape(-1, 1)
		if (copyY.size == 0 or copyYHat.size == 0 or copyY.ndim != 2 or copyYHat.ndim != 2 or copyY.shape != copyYHat.shape):
			return None
		return np.array([(copyY[i] - copyYHat[i]) ** 2 for i in range(copyY.shape[0])])

	def loss_(self, y, y_hat):
		ret = self.loss_elem_(y, y_hat)
		if (ret is None):
			return None
		return np.sum(ret) / (2 * y.shape[0])
	