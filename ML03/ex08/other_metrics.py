import numpy as np
import sklearn.metrics as skm

#!########################################################################################################!#
#!##############################################  FUNCTION  ##############################################!#
#!########################################################################################################!#

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


def accuracy_score_(y, y_hat):
	"""Compute the accuracy score.
	Args:
		y:a numpy.ndarray for the correct labels
		y_hat:a numpy.ndarray for the predicted labels
	Returns:
		The accuracy score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if (not isVector(y, -1) or not isVector(y_hat, y.shape[0])):
		return None
	tmpY = y.copy().reshape(-1)
	tmpY_hat = y_hat.copy().reshape(-1)
	return sum(tmpY == tmpY_hat) / tmpY.shape[0]

def precision_score_(y, y_hat, pos_label=1):
	"""Compute the precision score.
	Args:
		y:a numpy.ndarray for the correct labels
		y_hat:a numpy.ndarray for the predicted labels
		pos_label: str or int, the class on which to report the precision_score (default=1)
	Return:
		The precision score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if (not isVector(y, -1) or not isVector(y_hat, y.shape[0])):
		return None
	tmpY = y.copy().reshape(-1)
	tmpY_hat = y_hat.copy().reshape(-1)
	truePositive = sum((tmpY == pos_label) & (tmpY_hat == pos_label))
	falsePositive = sum((tmpY != pos_label) & (tmpY_hat == pos_label))
	return truePositive / (truePositive + falsePositive)

def recall_score_(y, y_hat, pos_label=1):
	"""Compute the recall score.
	Args:
		y:a numpy.ndarray for the correct labels
		y_hat:a numpy.ndarray for the predicted labels
		pos_label: str or int, the class on which to report the precision_score (default=1)
	Return:
		The recall score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if (not isVector(y, -1) or not isVector(y_hat, y.shape[0])):
		return None
	tmpY = y.copy().reshape(-1)
	tmpY_hat = y_hat.copy().reshape(-1)
	truePositive = sum((tmpY == pos_label) & (tmpY_hat == pos_label))
	falseNegative = sum((tmpY == pos_label) & (tmpY_hat != pos_label))
	return truePositive / (truePositive + falseNegative)

def f1_score_(y, y_hat, pos_label=1):
	"""Compute the f1 score.
	Args:
		y:a numpy.ndarray for the correct labels
		y_hat:a numpy.ndarray for the predicted labels
		pos_label: str or int, the class on which to report the precision_score (default=1)
	Returns:
		The f1 score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if (not isVector(y, -1) or not isVector(y_hat, y.shape[0])):
		return None
	precision = precision_score_(y, y_hat, pos_label)
	recall = recall_score_(y, y_hat, pos_label)
	return (2 * precision * recall) / (precision + recall)

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":

	y_hat1 = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
	y1 = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

	y_hat2 = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
	y2 = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

	ret1 = accuracy_score_(y1, y_hat1)
	ret2 = precision_score_(y1, y_hat1)
	ret3 = recall_score_(y1, y_hat1)
	ret4 = f1_score_(y1, y_hat1)

	ret5 = accuracy_score_(y2, y_hat2)
	ret6 = precision_score_(y2, y_hat2, pos_label='dog')
	ret7 = recall_score_(y2, y_hat2, pos_label='dog')
	ret8 = f1_score_(y2, y_hat2, pos_label='dog')

	ret9 = precision_score_(y2, y_hat2, pos_label='norminet')
	ret10 = recall_score_(y2, y_hat2, pos_label='norminet')
	ret11 = f1_score_(y2, y_hat2, pos_label='norminet')

	print(ret1)
	print(ret2)
	print(ret3)
	print(ret4)
	print(ret5)
	print(ret6)
	print(ret7)
	print(ret8)
	print(ret9)
	print(ret10)
	print(ret11)


	print("\n-------   Result expected   -------\n")
	print(0.5)
	print(0.4)
	print(0.6666666666666666)
	print(0.5)
	print(0.625)
	print(0.6)
	print(0.75)
	print(0.6666666666666665)
	print(0.6666666666666666)
	print(0.5)
	print(0.5714285714285715)


	print("\n-------   Sklearn Result    -------\n")
	print(skm.accuracy_score(y1, y_hat1))
	print(skm.precision_score(y1, y_hat1))
	print(skm.recall_score(y1, y_hat1))
	print(skm.f1_score(y1, y_hat1))

	print(skm.accuracy_score(y2, y_hat2))
	print(skm.precision_score(y2, y_hat2, pos_label='dog'))
	print(skm.recall_score(y2, y_hat2, pos_label='dog'))
	print(skm.f1_score(y2, y_hat2, pos_label='dog'))

	print(skm.precision_score(y2, y_hat2, pos_label='norminet'))
	print(skm.recall_score(y2, y_hat2, pos_label='norminet'))
	print(skm.f1_score(y2, y_hat2, pos_label='norminet'))

