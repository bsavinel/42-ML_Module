import numpy as np

#!#############################################################################################!#
#!########################################  Functions  ########################################!#
#!#############################################################################################!#

def add_intercept(x):
	"""Adds a column of 1’s to the non-empty numpy.array x.
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
	

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	x = np.arange(1,6)
	# x = [1, 2, 3, 4, 5]
	y = np.arange(1,10).reshape((3,3))
	# y = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
	ret1 = add_intercept(x)
	ret2 = add_intercept(y)
	print(ret1)
	print(ret2)
	print("\n-------   Result expected   -------\n")
	print(np.array([[1., 1.],
	[1., 2.],
	[1., 3.],
	[1., 4.],
	[1., 5.]]))
	print(np.array([[1., 1., 2., 3.],
	[1., 4., 5., 6.],
	[1., 7., 8., 9.]]))