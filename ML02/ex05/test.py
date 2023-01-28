import numpy as np
from mylinearregression import MyLinearRegression as MyLR

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLR(np.array([[1.], [1.], [1.], [1.], [1]]))

y_hat = mylr.predict_(X)
print(y_hat)
ret1 = mylr.loss_elem_(Y, y_hat)
print(ret1)
ret2 = mylr.loss_(Y, y_hat)
print(ret2)

mylr.alpha = 1.6e-4
mylr.max_iter = 200000

mylr.fit_(X, Y)
print(mylr.theta)
y_hat = mylr.predict_(X)
print(y_hat)
ret3 = mylr.loss_elem_(Y, y_hat)
print(ret3)
ret4 = mylr.loss_(Y, y_hat)
print(ret4)

print("\n-------   Result aproximative   -------\n")
print(np.array([[8.], [48.], [323.]]))
print(np.array([[225.], [0.], [11025.]]))
print(1875.0)
print(np.array([[18.188], [2.767], [-0.374], [1.392], [0.017]]))
print(np.array([[23.417], [47.489], [218.065]]))
print(np.array([[0.174], [0.260], [0.004]]))
print(0.0732)