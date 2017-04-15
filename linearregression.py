import numpy as np
import matplotlib.pyplot as plt

def best_fit(X, Y):
	a1 = ((X*Y).mean() - X.mean()*Y.mean())/((X**2).mean() - (X.mean())**2)
	a0 = Y.mean() - a1*X.mean()
	return a0, a1

X = np.array([6.32000000e-03  , 1.80000000e+01  , 2.31000000e+00  , 0.00000000e+00,
    5.38000000e-01  , 6.57500000e+00  , 6.52000000e+01  , 4.09000000e+00,
    1.00000000e+00 ,  2.96000000e+02 ,  1.53000000e+01  , 3.96900000e+02,
    4.98000000e+00])
Y = np.array([2.73100000e-02 ,  0.00000000e+00  , 7.07000000e+00 ,   0.00000000e+00,
    4.69000000e-01 ,  6.42100000e+00 ,  7.89000000e+01  , 4.96710000e+00 ,
    2.00000000e+00  ,  2.42000000e+02 ,  1.78000000e+01  , 3.96900000e+02,
    9.14000000e+00])

plt.scatter(X, Y)
a0,a1 = best_fit(X, Y)

X1 = 0
pred1 = a0 + a1 * X1
X2 = 400
pred2 = a0 + a1 * X2

X_Line = [X1, X2]
Y_Line = [pred1, pred2]
print(X_Line)
print(Y_Line)
plt.plot(X_Line, Y_Line)
plt.show()