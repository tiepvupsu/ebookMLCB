from __future__ import print_function
import numpy as np  
from math import ceil 
def softmax(Z):
    """
    Compute softmax values for each sets of scores in V.
    each column of V is a set of score.    
    """
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 0)
    return A

def softmax_stable(Z):
    """
    Compute softmax values for each sets of scores in Z.
    each row of Z is a set of scores.    
    """
    # Z = Z.reshape(Z.shape[0], -1)
    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

# cost or loss function  
def loss(X, y, W):
    """
    W: 2d numpy array of shape (d, C), 
        each column correspoding to one output node
    X: 2d numpy array of shape (N, d), each row is one data point
    y: 1d numpy array -- label of each row of X 
    """
    A = softmax_stable(X.dot(W))
    id1 = range(X.shape[0])
    return -np.mean(np.log(A[id1, y]))

# W_init = np.random.randn(d, C)

def grad(X, y, W):
    A = softmax_stable(X.dot(W))
    id1 = range(X.shape[0])
    A[id1, y] -= 1 # A - Y, shape of (N, C)
    return X.T.dot(A)/X.shape[0]
    
def numerical_grad(X, Y, W, loss):
    eps = 1e-6
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy()
            W_n = W.copy()
            W_p[i, j] += eps 
            W_n[i, j] -= eps
            g[i,j] = (loss(X, Y, W_p) - loss(X, Y, W_n))/(2*eps)
    return g 

def softmax_fit(X, y, W, lr = 0.01, nepoches = 100, tol = 1e-5, batch_size = 10):
    W_old = W.copy()
    ep = 0 
    loss_hist = [loss(X, y, W)]
    N = X.shape[0]
    nbatches = int(np.ceil(float(N)/batch_size))
    while ep < nepoches: 
        ep += 1 
        mix_ids = np.random.permutation(N)
        for i in range(nbatches):
            batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), N)] 
            # print(batch_ids)
            X_batch, y_batch = X[batch_ids], y[batch_ids]
            W -= lr*grad(X_batch, y_batch, W)
        loss_hist.append(loss(X, y, W))
        if np.linalg.norm(W - W_old)/W.size < tol:
            break 
        W_old = W.copy()
    return W, loss_hist 





d = 100
C = 3 
N = 3000
X = np.random.randn(N, d)
y = np.random.randint(0, C, N) 
W = np.random.randn(d, C) 
# print(y.shape)
# print(loss(X, y, W))
# g1 = grad(X, y, W)
# g2 = numerical_grad(X, y, W, loss)

# # print(g1)
# # print(g2)
# print(np.linalg.norm(g1 - g2)/g1.size)

W, loss_hist = softmax_fit(X, y, W, batch_size = 100)
import matplotlib.pyplot as plt

print(loss_hist[-1])
plt.plot(loss_hist)
plt.show() 