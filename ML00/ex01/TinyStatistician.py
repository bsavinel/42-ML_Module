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
			return [tmp[int(len(tmp) / 4)], tmp[int(len(tmp) * (3 / 4))]]
		else:
			return None
		
#!---------------- percentile ----------------!#

	def percentile(self, x, percentile):
		if check_type(x) and isinstance(percentile, (int, float)) and percentile > 0 and percentile < 100:
			tmp = x.copy()
			tmp.sort()
			place = (percentile * (len(tmp) - 1)) / 100
			if ((place + 1) % 1 == 0):
				return tmp[m.floor(place)]
			else:
				return (tmp[m.ceil(place)] * (place - m.floor(place))) + (tmp[m.floor(place)] * (m.ceil(place) - place))
		else:
			return None

#!---------------- Variance ----------------!#

	def var(self, x):
		if check_type(x):
			moy = self.mean(x)
			somme = 0
			for i in x:
				somme += (i - moy) ** 2
			return somme / (len(x) - 1)
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
	print("---------- Mean ----------")
	print(ts.mean(x))
	print("---------- Median ----------")
	print(ts.median(x))
	print("---------- Quartile ----------")
	print(ts.quartile(x))
	print("---------- Percentile ----------")
	print("10 :", ts.percentile(x, 10))
	print("15 :", ts.percentile(x, 15))
	print("20 :", ts.percentile(x, 20))
	print("---------- Variance ----------")
	print(ts.var(x))
	print("---------- Standard Deviation ----------")
	print(ts.std(x))
	print("\n##############################")
	print("########## EXPECTED ##########")
	print("##############################\n")
	print("---------- Mean ----------")
	print(82.4)
	print("---------- Median ----------")
	print(42.0)
	print("---------- Quartile ----------")
	print([10.0, 59.0])
	print("---------- Percentile ----------")
	print("10 :", 4.6)
	print("15 :", 6.4)
	print("20 :", 8.2)
	print("---------- Variance ----------")
	print(15349.3)
	print("---------- Standard Deviation ----------")
	print(123.89229193133849)