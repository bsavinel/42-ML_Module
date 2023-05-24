import numpy as np

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

def simple_gradient(x, y, theta):
	if (not check_matrix(x, -1, 1) or not check_matrix(y, x.shape[0], 1) or not check_matrix(theta, 2, 1)):
		return None
	copyX = np.insert(x, 0, 1, axis=1)
	transpX = copyX.transpose()
	return (transpX @ (copyX @ theta - y)) / x.shape[0]

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
		if (not check_matrix(x, -1, 1) or not check_matrix(y, x.shape[0], 1)):
			return None
		for i in range(self.max_iter):
			self.thetas = self.thetas - (self.alpha * simple_gradient(x, y, self.thetas))

	def predict_(self, x):
		if (not check_matrix(x, -1, 1)):
			return None
		return np.dot(add_intercept(x), self.thetas)

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
	