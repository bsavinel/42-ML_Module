import numpy as np

def check_matrix(m, sizeX, sizeY, dim = 2):
	if (not isinstance(m, np.ndarray)):
		return False
	if (m.ndim != dim or m.size == 0):
		return False
	if (sizeX != -1 and m.shape[0] != sizeX):
		return False
	if (dim == 2 and sizeY != -1 and m.shape[1] != sizeY):
		return False
	return True

def sigmoid_(x):
	"""
	Compute the sigmoid of a vector.
	Args:
		x: has to be a numpy.ndarray of shape (m, 1).
	Returns:
		The sigmoid value as a numpy.ndarray of shape (m, 1).
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, 1)):
		return None
	return 1 / (1 + np.exp(x * -1))	


class MyLogisticRegression():
	"""
	Description:
		My personnal logistic regression to classify things.
	"""
	def __init__(self, theta, alpha=0.001, max_iter=1000):
		if ((not isinstance(alpha, float)) or (not isinstance(max_iter, int)) or max_iter < 0 or not check_matrix(theta, -1, 1)):
			raise ValueError 
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta

	def predict_(self, x):
		if (not check_matrix(x, -1, self.theta.shape[0] - 1)):
			return None
		copyx = np.insert(x, 0, 1, axis = 1)
		return sigmoid_(np.dot(copyx, self.theta))

	def loss_elem_(self, y, y_hat, eps=1e-15):
		if (not check_matrix(y, -1, 1) or not check_matrix(y_hat, y.shape[0], 1) or not isinstance(eps, float)):
			return None
		copyYHat = y_hat.copy()
		copyYHat = np.where(copyYHat == 1, copyYHat - eps, copyYHat)
		copyYHat = np.where(copyYHat == 0, copyYHat + eps, copyYHat)
		return y * np.log(copyYHat) + (1 - y) * np.log(1 - copyYHat)

	def loss_(self, y, y_hat, eps=1e-15):
		if (not check_matrix(y, -1, 1) or not check_matrix(y_hat, y.shape[0], 1) or not isinstance(eps, float)):
			return None
		copyYHat = y_hat.copy()
		copyYHat = np.where(copyYHat == 1, copyYHat - eps, copyYHat)
		copyYHat = np.where(copyYHat == 0, copyYHat + eps, copyYHat)
		vec_one = np.ones(y.shape).reshape((-1, 1))
		dot1 = np.dot(y.reshape(-1), np.log(copyYHat).reshape(-1))
		dot2 = np.dot((vec_one - y).reshape(-1), np.log(vec_one - copyYHat).reshape(-1))
		return (dot1 + dot2) / (-y.shape[0])

	def log_gradient(self, x, y):
		if (not check_matrix(x, -1, self.theta.shape[0] - 1) or not check_matrix(y, x.shape[0], 1)):
			return None
		copyX = np.insert(x, 0, 1, axis = 1)
		return copyX.T @ (self.predict_(x) - y) / x.shape[0]

	def fit_(self, x, y):
		if (not check_matrix(x, -1, self.theta.shape[0] - 1) or not check_matrix(y, x.shape[0], 1)):
			return None
		self.theta = self.theta.astype('float64')
		copyX = x.copy().astype('float64')
		copyY = y.copy().astype('float64')
		for i in range(self.max_iter):
			self.theta = self.theta - (self.alpha * self.log_gradient(copyX, copyY))
