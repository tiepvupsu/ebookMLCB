# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(2)

X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# extened data 
X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)

def sigmoid(s):
    return 1/(1 + np.exp(-s))

# def grad_sigmoid(s):
#     return sigmoid(s)*(1 - sigmoid(s))

# def loss()

def logistic_sigmoid_regression(X, y, w_init, eta):
    w = [w_init]
    max_iter = 100
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    while it < max_iter:
        it += 1
        # mix data 
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            w.append(w_new)
    return w 

eta =  .05
d = X.shape[0]
w_init = np.random.randn(d, 1)

w = logistic_sigmoid_regression(X, y, w_init, eta)
print(w)

print(sigmoid(np.dot(w[-1].T, X)))
