import numpy as np
import pandas as pd
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

def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
	"""Compute confusion matrix to evaluate the accuracy of a classification.
	Args:
		y:a numpy.array for the correct labels
		y_hat:a numpy.array for the predicted labels
		labels: optional, a list of labels to index the matrix.
			This may be used to reorder or select a subset of labels. (default=None)
		df_option: optional, if set to True the function will return a pandas DataFrame
			instead of a numpy array. (default=False)
	Return:
		The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
		None if any error.
	Raises:
		This function should not raise any Exception.
	"""
	if (not isVector(y_true, -1) or not isVector(y_hat, y_true.shape[0])):
		return None
	tmpY = y_true.copy().reshape(-1)
	tmpY_hat = y_hat.copy().reshape(-1)
	if (labels is None):
		labelsCopy = np.unique(np.concatenate((tmpY, tmpY_hat)).reshape(-1))
	elif (isinstance(labels, list)):
		labelsCopy = np.array(labels)
	elif (not isVector(labels)):
		return None
	else:
		labelsCopy =  labels.copy().reshape(-1)
	ret = np.zeros((labelsCopy.shape[0], labelsCopy.shape[0])).astype(int)
	# for i in range(tmpY.shape[0]):
	# 	ret[np.where(labelsCopy == tmpY[i]), np.where(labelsCopy == tmpY_hat[i])] += 1
	for i in range(labelsCopy.shape[0]):
		for j in range(labelsCopy.shape[0]):
			ret[i, j] = sum((tmpY == labelsCopy[i]) & (tmpY_hat == labelsCopy[j]))
	if (df_option):
		return pd.DataFrame(ret, index=labelsCopy, columns=labelsCopy)
	return ret

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
	y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])
	ret1 = confusion_matrix_(y, y_hat)
	ret2 = confusion_matrix_(y, y_hat, labels=['dog', 'norminet'])
	ret3 = confusion_matrix_(y, y_hat, df_option=True)
	ret4 = confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True)
	print(ret1)
	print(ret2)
	print(ret3)
	print(ret4)
	print("\n-------   Result expected   -------\n")
	print(np.array([[0, 0, 0],[0, 2, 1],[1, 0, 2]]))
	print(np.array([[2, 1],[0, 2]]))
	print("""         bird dog norminet
bird        0   0        0
dog         0   2        1
norminet    1   0        2""")
	print("""      bird dog
bird     0   0
dog      0   2""")
	print("\n-------   Sklearn Result    -------\n")
	print(skm.confusion_matrix(y, y_hat))
	print(skm.confusion_matrix(y, y_hat, labels=['dog', 'norminet']))
	print("Dosen't exist")
	print("Dosen't exist")