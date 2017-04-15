import numpy as np
import matplotlib.pyplot as plt

def stepGradient(b_current, m_current, X,Y, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(X))
    for i in range(0, len(X)):
        x = X[i]
        y = Y[i]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    plt.plot(new_b,new_m)
    print(new_b,new_m)
    
    return [new_b, new_m]

def gradient_descent_runner(X,Y, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = stepGradient(b, m, X,Y, learning_rate)
        
    plt.show()
    return [b, m]


def best_fit(X, Y):
	a1 = ((X*Y).mean() - X.mean()*Y.mean())/((X**2).mean() - (X.mean())**2)
	a0 = Y.mean() - a1*X.mean()
	return a0, a1

X = np.array([1,2,3,4,5,6])
Y = np.array([1.5,4,4,6,7,8])

plt.scatter(X, Y)
a0,a1 = gradient_descent_runner(X,Y,0,0,0.001,100)

X1 = 0
pred1 = a0 + a1 * X1
X2 = 500
pred2 = a0 + a1 * X2

X_Line = [X1, X2]
Y_Line = [pred1, pred2]
print(X_Line)
print(Y_Line)
#plt.plot(X_Line, Y_Line)
#plt.show()