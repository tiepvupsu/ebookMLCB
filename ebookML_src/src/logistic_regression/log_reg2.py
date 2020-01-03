from __future__ import print_function
import numpy as np 


def sigmoid(S):
    """
    S: an numpy array
    return sigmoid function of each element of S
    """
    return 1/(1 + np.exp(-S))

def bias_trick(X):
    N = X.shape[0]
    return np.concatenate((np.ones((N, 1)), X), axis = 1)

def prob(w, X):
        return sigmoid(X.dot(w))

def predict(w, X, threshold = 0.5):
    """
    predict output of each row of X
    X: a numpy array of shape
    threshold: a threshold between 0 and 1 
    """
    res = np.zeros(X.shape[0])
    res[np.where(prob(w, X) > threshold)[0]] = 1
    return res 

def loss(w, X, y):
    # if self.bias: X = bias_trick(X)
    z = prob(w, X)
    return -np.mean(y*np.log(z) + (1-y)*np.log(1-z))

def logistic_regression(w_init, X, y, lr = 0.1, nepoches = 100):
    N = X.shape[0]
    d = X.shape[1]
    w = w_old = w_init 
    loss_hist = [loss(w_init, X, y)]
    ep = 0 
    while ep < nepoches: 
        ep += 1
        mix_ids = np.random.permutation(N)
        for i in mix_ids:
            xi = X[i]
            yi = y[i]
            zi = sigmoid(xi.dot(w))
            w = w + lr*(yi - zi)*xi 
        loss_hist.append(loss(w, X, y))
        if np.linalg.norm(w - w_old)/d < 1e-4:
            break 
    return w, loss_hist 

np.random.seed(2)
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

X = bias_trick(X)

w_init = np.random.randn(X.shape[1])
w, loss_hist = logistic_regression(w_init, X, y)
print(prob(w, X))
print(predict(w, X))

















