import numpy as np
from mylinearregression import MyLinearRegression as mlr, check_matrix

class MyRidge(mlr):
	"""
	Description:
	My personnal ridge regression class to fit like a boss.
	"""
	def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
		if (not check_matrix(thetas, -1, 1) or not isinstance(lambda_, float) or not isinstance(alpha, float) or not isinstance(max_iter, int)):
			return None
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas
		self.lambda_ = lambda_
		self.loss_evolution = []

	def get_params_(self):
		return {"thetas": self.thetas, "alpha": self.alpha, "max_iter": self.max_iter, "lambda_": self.lambda_}
	
	def set_params_(self, **kwargs):
		if (not isinstance(kwargs, dict)):
			return None
		for name in kwargs:
			if (name == "thetas"):
				if (check_matrix(kwargs[name], -1, 1)):
					self.thetas = kwargs[name]
			elif (name == "alpha"):
				if (isinstance(kwargs[name], float)):
					self.alpha = kwargs[name]
			elif (name == "max_iter"):
				if (isinstance(kwargs[name], int)):
					self.max_iter = kwargs[name]
			elif (name == "lambda_"):
				if (isinstance(kwargs[name], float)):
					self.lambda_ = kwargs[name]

	def predict_(self,x):
		if (not check_matrix(x, -1, self.thetas.shape[0] - 1)):
			return None
		Xprime = np.insert(x, 0, 1, axis=1)
		return Xprime @ self.thetas

	def gradient_(self, x, y):
		if (not check_matrix(x, -1, self.thetas.shape[0] - 1) or not check_matrix(y, x.shape[0], 1)):
			print("x", x.shape)
			print("y", y.shape)
			print("thetas", self.thetas.shape)
			return None
		newTheta = np.copy(self.thetas).reshape(-1,1)
		newTheta[0][0] = 0
		Xprime = np.insert(x, 0, 1, axis=1)
		pred = self.predict_(x)
		return (np.matmul(Xprime.T,(pred - y)) + (self.lambda_ * newTheta)) / y.shape[0]
	
	def loss_(self,y, y_hat):
		if (not check_matrix(y, -1, 1) or not check_matrix(y_hat, y.shape[0], 1)):
			return None
		sub = y_hat.reshape(-1) - y.reshape(-1)
		return (np.dot(sub,sub) + (self.lambda_ * self.l2(self.thetas))) / (2 * y.shape[0])
	
	def fit_(self, x, y):
		if (not check_matrix(x, -1, self.thetas.shape[0] - 1) or not check_matrix(y, x.shape[0], 1)):
			return None
		self.thetas = np.copy(self.thetas.astype('float64'))
		for i in range(self.max_iter):
			self.loss_evolution.append(self.loss_(y, self.predict_(x)))
			self.thetas = self.thetas - (self.alpha * self.gradient_(x, y))

	def loss_elem_(self, y, y_hat):
		if (not check_matrix(y, -1, 1) or not check_matrix(y_hat, y.shape[0], 1)):
			return None
		return np.array([(y[i] - y_hat[i]) ** 2 + self.lambda_ * self.l2(self.thetas) for i in range(y.shape[0])])

	@staticmethod
	def l2(theta):
		if (not check_matrix(theta, -1, 1)):
			return None
		newTheta = np.copy(theta).reshape(-1)
		newTheta[0] = 0
		return np.dot(newTheta, newTheta)
	
	