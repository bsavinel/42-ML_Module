import numpy as np
import math as m

#!#####################################################################################################!#
#!########################################  TINY STATISTICIAN  ########################################!#
#!#####################################################################################################!#

def check_type(x):
	if (isinstance(x, list) or isinstance(x, np.ndarray)):
		for i in x:
			if (not isinstance(i, (int, float))):
				return False
		return True
	else:
		return False

class TinyStatistician():

	def __init__(self):
		pass

#!---------------- Mean ----------------!#

	def mean(self, x):
		if check_type(x):
			somme = 0
			for i in x:
				somme += i
			return somme / len(x)
		else:
			return None

#!---------------- Median ----------------!#

	def median(self, x):
		if check_type(x):
			tmp = x.copy()
			tmp.sort()
			if (len(tmp) % 2 == 1):
				return tmp[int(len(tmp) / 2)]
			else:
				return (tmp[int(len(tmp) / 2)] + tmp[int(len(tmp) / 2) + 1]) / 2
		else:
			return None

#!---------------- Quartile ----------------!#

	def quartile(self, x):
		if check_type(x):
			tmp = x.copy()
			tmp.sort()
			if (len(x) % 4 == 0):
				return [tmp[int(len(tmp) / 4)], tmp[int(len(tmp) * (3 / 4))]]
			else:
				return [(tmp[m.floor(len(tmp) / 4)] + tmp[m.ceil(len(tmp) / 4)]) / 2,
						(tmp[m.floor(len(tmp) * (3 / 4))] + tmp[m.ceil(len(tmp) * (3 / 4))]) / 2]
		else:
			return None
		
#!---------------- percentile ----------------!#

	def percentile(self, x, percentile):
		if check_type(x) and isinstance(percentile, (int, float)) and percentile > 0 and percentile < 100:
			tmp = x.copy()
			tmp.sort()
			return (tmp[m.floor(len(tmp) / (percentile / 100))] + tmp[m.ceil(len(tmp) / (percentile / 100))]) / 2,
		else:
			return None

#!---------------- Variance ----------------!#

	def var(self, x):
		if check_type(x):
			moy = self.mean(x)
			somme = 0
			for i in x:
				somme += (i - moy) ** 2
		else:
			return None

#!---------------- Standard Deviation ----------------!#

	def std(self, x):
		if check_type(x):
			return m.sqrt(self.var(x))
		else:
			return None

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	ts = TinyStatistician()
	x = [1, 42, 300, 10, 59]
	print(ts.mean(x))
	print(ts.median(x))
	print(ts.quartile(x))
	print(ts.percentile(x, 25))
	print(ts.var(x))
	print(ts.std(x))