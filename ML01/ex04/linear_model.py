import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#


#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#


if __name__ == "__main__":
	data = pd.read_csv("are_blue_pills_magics.csv")
	Xpill = np.array(data["Micrograms"]).reshape(-1,1)
	Yscore = np.array(data["Score"]).reshape(-1,1)
	linear_model1 = MyLR(np.array([[89.0], [-8]]))
	linear_model2 = MyLR(np.array([[89.0], [-6]]))
	Y_model1 = linear_model1.predict_(Xpill)
	Y_model2 = linear_model2.predict_(Xpill)
	Yscore = Yscore.reshape(-1)
	Y_model1 = Y_model1.reshape(-1)
	Y_model2 = Y_model2.reshape(-1)
	print(MyLR.mse_(Yscore, Y_model1))
	print(MyLR.mse_(Yscore, Y_model2))
	print("\n-------   Result with sklearn   -------\n")
	print(mean_squared_error(Yscore, Y_model1))
	print(mean_squared_error(Yscore, Y_model2))
	print("\n-------   Result expected   -------\n")
	print(57.60304285714282)
	print(232.16344285714285)