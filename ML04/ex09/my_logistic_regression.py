import numpy as np

def check_matrix(m, sizeX = -1, sizeY = -1, dim = 2):
	if (not isinstance(m, np.ndarray)):
		return False
	if (m.ndim != dim or m.size == 0):
		return False
	if (sizeX != -1 and m.shape[0] != sizeX):
		return False
	if (dim == 2 and sizeY != -1 and m.shape[1] != sizeY):
		return False
	return True

def isVector(m, sizeX = -1):
	if(not check_matrix(m, sizeX, 1) and not check_matrix(m, sizeX, dim=1)):
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

	supported_penalities = ['l2']
	def __init__(self, theta, alpha=0.001, max_iter=1000, penality='l2', lambda_=1.0):
		if ((not isinstance(alpha, float)) or (not isinstance(max_iter, int)) or max_iter < 0 or not check_matrix(theta, -1, 1)):
			raise ValueError 
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta
		self.penality = penality
		self.lambda_ = lambda_ if penality in self.supported_penalities else 0
		self.loss_evolution = []

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

	def no_reg_loss_(self, y, y_hat, eps=1e-15):
		if (not check_matrix(y, -1, 1) or not check_matrix(y_hat, y.shape[0], 1) or not isinstance(eps, float)):
			return None
		copyYHat = y_hat.copy()
		copyYHat = np.where(copyYHat == 1, copyYHat - eps, copyYHat)
		copyYHat = np.where(copyYHat == 0, copyYHat + eps, copyYHat)
		vec_one = np.ones(y.shape).reshape((-1, 1))
		dot1 = np.dot(y.reshape(-1), np.log(copyYHat).reshape(-1))
		dot2 = np.dot((vec_one - y).reshape(-1), np.log(vec_one - copyYHat).reshape(-1))
		return (dot1 + dot2) / (-y.shape[0])
	
	def loss_(self, y, y_hat):
		if (self.penality != 'l2'):
			return self.no_reg_loss_(y, y_hat)
		if (not check_matrix(y, -1, 1) or not check_matrix(y_hat, y.shape[0], 1)):
			return None
		return self.no_reg_loss_(y, y_hat) + self.lambda_ / (2 * y.shape[0]) * self.l2(self.theta)

	def gradient_(self, x, y):
		if (not check_matrix(x, -1 , self.theta.shape[0] - 1) or not check_matrix(y, x.shape[0], 1)):
			return None
		newTheta = np.copy(self.theta).reshape(-1,1)
		newTheta[0][0] = 0
		Xprime = np.insert(x, 0, 1, axis=1)
		pred = sigmoid_(Xprime @ self.theta)
		return ((Xprime.T @ (pred - y)) + (self.lambda_ * newTheta)) / y.shape[0]

	def fit_(self, x, y):
		if (not check_matrix(x, -1, self.theta.shape[0] - 1) or not check_matrix(y, x.shape[0], 1)):
			return None
		self.theta = self.theta.astype('float64')
		copyX = x.copy().astype('float64')
		copyY = y.copy().astype('float64')
		self.loss_evolution.append(self.loss_(y, self.predict_(x)))
		for i in range(self.max_iter):
			self.theta = self.theta - (self.alpha * self.log_gradient(copyX, copyY))
			self.loss_evolution.append(self.loss_(y, self.predict_(x)))

	@staticmethod
	def l2(theta):
		if (not check_matrix(theta, -1, 1)):
			return None
		newTheta = np.copy(theta).reshape(-1)
		newTheta[0] = 0
		return np.dot(newTheta, newTheta)
	
	@staticmethod
	def precision_score_(y, y_hat, pos_label=1):
		if (not isVector(y, -1) or not isVector(y_hat, y.shape[0])):
			return None
		tmpY = y.copy().reshape(-1)
		tmpY_hat = y_hat.copy().reshape(-1)
		truePositive = sum((tmpY == pos_label) & (tmpY_hat == pos_label))
		falsePositive = sum((tmpY != pos_label) & (tmpY_hat == pos_label))
		if (truePositive == 0 and falsePositive != 0):
			return 0.0
		elif (falsePositive == 0):
			return 1.0
		return truePositive / (truePositive + falsePositive)

	@staticmethod
	def recall_score_(y, y_hat, pos_label=1):
		if (not isVector(y, -1) or not isVector(y_hat, y.shape[0])):
			return None
		tmpY = y.copy().reshape(-1)
		tmpY_hat = y_hat.copy().reshape(-1)
		truePositive = sum((tmpY == pos_label) & (tmpY_hat == pos_label))
		falseNegative = sum((tmpY == pos_label) & (tmpY_hat != pos_label))
		if (truePositive == 0 and falseNegative != 0):
			return 0.0
		elif (falseNegative == 0):
			return 1.0
		return truePositive / (truePositive + falseNegative)

	@staticmethod	
	def f1_score_(y, y_hat, pos_label=1):
		if (not isVector(y, -1) or not isVector(y_hat, y.shape[0])):
			return None
		precision = MyLogisticRegression.precision_score_(y, y_hat, pos_label)
		recall = MyLogisticRegression.recall_score_(y, y_hat, pos_label)
		if (precision == 0 or recall == 0):
			return 0.0
		return (2 * precision * recall) / (precision + recall)
