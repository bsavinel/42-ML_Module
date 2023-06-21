import numpy as np
from my_logistic_regression import MyLogisticRegression as mylogr

x = np.array([[0, 2, 3, 4],[2, 4, 5, 5],[1, 3, 2, 7]])
y = np.array([[0], [1], [1]])
theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
test1 = mylogr(theta)
ret1 = test1.gradient_(x, y)
mylogr.lambda_ = 0.5
ret2 = test1.gradient_(x, y)
mylogr.lambda_ = 0.0
ret3 = test1.gradient_(x, y)
print(ret1)
print(ret2)
print(ret3)

y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
test2 = mylogr(theta,lambda_=0.5)
ret4 = test2.loss_(y, y_hat)
test2.lambda_ = 0.05
ret5 = test2.loss_(y, y_hat)
test2.lambda_ = 0.9
ret6 = test2.loss_(y, y_hat)
print(ret4)
print(ret5)
print(ret6)

theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
model1 = mylogr(theta, lambda_=5.0)
print(model1.penality)
print(model1.lambda_)
model2 = mylogr(theta, penality=None)
print(model2.penality)
print(model2.lambda_)
model3 = mylogr(theta, penality=None, lambda_=2.0)
print(model3.penality)
print(model3.lambda_)
model = mylogr(theta, lambda_=5.0)

print("\n-------   Result expected   -------\n")
print(np.array([[-0.55711039],[-1.40334809],[-1.91756886],[-2.56737958],[-3.03924017]]))
print(np.array([[-0.55711039],[-1.15334809],[-1.96756886],[-2.33404624],[-3.15590684]]))
print(np.array([[-0.55711039],[-0.90334809],[-2.01756886],[-2.10071291],[-3.27257351]]))
print(0.43377043716475955)
print(0.13452043716475953)
print(0.6997704371647596)
print('l2')
print(5.0)
print(None)
print(0.0)
print(None)
print(0.0)
