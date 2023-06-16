import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

def fistPart(Sell_price, other, theta ,alpha, max_iter, xlabel):
	linearModel = MyLR(theta, alpha, max_iter)
	linearModel.fit_(other, Sell_price)
	plt.plot(other, Sell_price, 'bo', label="Sell price")
	plt.plot(other, linearModel.predict_(other), 'r.', label="Predicted sell price")
	plt.xlabel(xlabel)
	plt.ylabel("y: sell price (in keuros)")
	plt.grid()
	plt.legend()
	plt.show()

def secondPart(Sell_price, other, predict, xlabel):
	plt.plot(other, Sell_price, 'bo', label="Sell price")
	plt.plot(other, predict, 'r.', label="Predicted sell price")
	plt.xlabel(xlabel)
	plt.ylabel("y: sell price (in keuros)")
	plt.grid()
	plt.legend()
	plt.show()


#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == "__main__":
	data = pd.read_csv("spacecraft_data.csv")
	Sell_price = np.array(data["Sell_price"]).reshape(-1,1)
	Age = np.array(data["Age"]).reshape(-1,1)
	Thrust_power = np.array(data["Thrust_power"]).reshape(-1,1)
	Terameters = np.array(data["Terameters"]).reshape(-1,1)

	fistPart(Sell_price, Age, np.array([[650], [-1]]), 2.5e-5, 200000, "x1: age (in years)")
	fistPart(Sell_price, Thrust_power, np.array([[60], [-1]]), 5.5e-5, 200000, "x2: thrust power (in 10Km/s)")
	fistPart(Sell_price, Terameters, np.array([[200], [12]]), 7.5e-5, 200000, "x3: distance totalizer value of spacecraft (in Tmeters)")

	myLR_age = MyLR(np.array([[1000.0], [-1.0]]), alpha = 2.5e-5, max_iter = 100000)
	myLR_age.fit_(Age[:,0].reshape(-1,1), Sell_price)
	y_pred = myLR_age.predict_(Age[:,0].reshape(-1,1))
	print(myLR_age.mse_(y_pred,Sell_price))

	data = pd.read_csv("spacecraft_data.csv")
	X = np.array(data[["Age","Thrust_power","Terameters"]])
	Y = np.array(data[["Sell_price"]])
	my_lreg = MyLR(np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1),  alpha = 1e-5, max_iter = 600000)
	print(my_lreg.mse_(Y, my_lreg.predict_(X)))
	my_lreg.fit_(X,Y)
	print(my_lreg.theta)
	print(my_lreg.mse_(Y, my_lreg.predict_(X)))

	predict = my_lreg.predict_(X)
	secondPart(Sell_price, Age, predict, "x1: age (in years)")
	secondPart(Sell_price, Thrust_power, predict, "x2: thrust power (in 10Km/s)")
	secondPart(Sell_price, Terameters, predict, "x3: distance totalizer value of spacecraft (in Tmeters)")

	print("\n-------   Aproximative result   -------\n")
	print(55736.86719)
	print(144044.877)
	print(np.array([[334.994],[-22.535],[5.857],[-2.586]]))
	print(586.896999)