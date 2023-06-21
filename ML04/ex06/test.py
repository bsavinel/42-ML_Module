from ridge import MyRidge
import numpy as np

y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
Mlr = MyRidge(theta)

Mlr.set_params_(lambda_ = 0.5)
ret0 = Mlr.loss_elem_(y, y_hat)
ret1 = Mlr.loss_(y, y_hat)
Mlr.set_params_(lambda_ = 0.05)
ret2 = Mlr.loss_(y, y_hat)
Mlr.set_params_(lambda_ = 0.9)
ret3 = Mlr.loss_(y, y_hat)

x = np.array([[ -6, -7, -9],[ 13, -2, 14],[ -7, 14, -1],[ -8, -4, 6],[ -5, -9, 6],[ 1, -5, 11],[ 9, -11, 8]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
theta = np.array([[7.01], [3], [10.5], [-6]])
Mlr = MyRidge(theta)
Mlr.set_params_(lambda_ = 1.)
ret4 = Mlr.gradient_(x, y)
Mlr.set_params_(lambda_ = 0.5)
ret5 = Mlr.gradient_(x, y)
Mlr.set_params_(lambda_ = 0.0)
ret6 = Mlr.gradient_(x, y)
print(ret0)
print(ret1)
print(ret2)
print(ret3)
print(ret4)
print(ret5)
print(ret6)
print("\n-------   Result expected   -------\n")
print(0.8503571428571429)
print(0.5511071428571429)
print(1.116357142857143)
print(np.array([[ -60.99 ],[-195.64714286],[ 863.46571429],[-644.52142857]]))
print(np.array([[ -60.99 ],[-195.86142857],[ 862.71571429],[-644.09285714]]))
print(np.array([[ -60.99 ],[-196.07571429],[ 861.96571429],[-643.66428571]]))