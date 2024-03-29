import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
Y = np.array([[1], [0], [1]])
thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
mylr = MyLR(thetas)
ret = mylr.predict_(X)
print(ret)
print(mylr.loss_(Y, ret))
mylr.fit_(X, Y)
print(mylr.theta)
ret2 = mylr.predict_(X)
print(ret2)
print(mylr.loss_(Y, ret2))
print("\n-------   Result expected   -------\n")
print(np.array([[0.99930437],[1. ],[1. ]]))
print(11.513157421577004)
print(np.array([[2.11826435],[0.10154334],[6.43942899],[-5.10817488],[0.6212541]]))
print(np.array([[0.57606717],[0.68599807],[0.06562156]]))
print(1.4779126923052268)