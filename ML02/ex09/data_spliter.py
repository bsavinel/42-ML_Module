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

def unison_shuffled_copies(a, b):
	"""Shuffle two arrays in the same way."""
	if (len(a) != len(b) or len(a) == 0):
		return None
	p = np.random.permutation(len(a))
	return a[p], b[p]

def data_spliter(x, y, proportion):
	"""Shuffles and splits the dataset (given by x and y) into a training and a test set,
	while respecting the given proportion of examples to be kept in the training set.
	Args:
		x: has to be an numpy.array, a matrix of dimension m * n.
		y: has to be an numpy.array, a vector of dimension m * 1.
		proportion: has to be a float, the proportion of the dataset that will be assigned to the
		training set.
	Return:
		(x_train, x_test, y_train, y_test) as a tuple of numpy.array
		None if x or y is an empty numpy.array.
		None if x and y do not share compatible dimensions.
		None if x, y or proportion is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if (not check_matrix(x, -1, -1) or not check_matrix(y, x.shape[0], 1) or proportion > 1 or proportion < 0):
		return None
	index_prop = int(x.shape[0] * proportion)
	copyX, copyY = unison_shuffled_copies(x, y)
	return (copyX[:index_prop], copyX[index_prop:], copyY[:index_prop], copyY[index_prop:])


#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
	x2 = np.array([[ 1, 42],[300, 10],[ 59, 1],[300, 59],[ 10, 42]])
	y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

	ret1 = data_spliter(x1, y, 0.8)
	ret2 = data_spliter(x1, y, 0.5)
	ret3 = data_spliter(x2, y, 0.8)
	ret4 = data_spliter(x2, y, 0.5)
	print("Exemple 1 :", ret1)
	print("Exemple 2 :", ret2)
	print("Exemple 3 :", ret3)
	print("Exemple 4 :", ret4)
	print("\n-------   Result expected   -------\n")
	print("Exemple 1 :", (np.array([ 1, 59, 42, 300]).reshape((-1,1)), np.array([10]).reshape((-1,1)), np.array([0, 0, 1, 0]).reshape((-1,1)), np.array([1]).reshape((-1,1))))
	print("Exemple 2 :", (np.array([59, 10]).reshape((-1,1)), np.array([ 1, 300, 42]).reshape((-1,1)), np.array([0, 1]).reshape((-1,1)), np.array([0, 0, 1]).reshape((-1,1))))
	print("Exemple 3 :", (np.array([[ 10, 42],[300, 59],[ 59, 1],[300, 10]]),np.array([[ 1, 42]]),np.array([0, 1, 0, 1]).reshape((-1,1)),np.array([0]).reshape((-1,1))))
	print("Exemple 4 :", (np.array([[59, 1],[10, 42]]),np.array([[300, 10],[300, 59],[ 1, 42]]),np.array([0, 0]).reshape((-1,1)),np.array([1, 1, 0]).reshape((-1,1))))