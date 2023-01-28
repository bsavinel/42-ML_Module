import numpy as np

def minmax(x):
	"""Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
	Args:
		x: has to be an numpy.ndarray, a vector.
	Returns:
		x' as a numpy.ndarray.
		None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
	Raises:
		This function shouldn't raise any Exception.
	"""
	if (not isinstance(x, np.ndarray) or x.size == 0):
		return None
	if (x.ndim != 1 and not (x.ndim == 2 and x.shape[1] == 1)):
		return None
	Xcopy = x.reshape(-1)
	min = np.min(Xcopy)
	max = np.max(Xcopy)
	return (Xcopy - min) / (max - min)

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#


if __name__ == "__main__":
	X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
	Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	ret1 = minmax(X)
	ret2 = minmax(Y)
	print(ret1)
	print(ret2)
	print("\n-------   Result expected   -------\n")
	print(np.array([0.58333333, 1. , 0.33333333, 0.77777778, 0.91666667,0.66666667, 0. ]))
	print(np.array([0.63636364, 1. , 0.18181818, 0.72727273, 0.93939394,0.6969697 , 0. ]))