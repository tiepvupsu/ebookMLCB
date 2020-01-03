
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(4)

means = [[-1, -1], [1, -1], [0, 1]]
cov = [[1, 0], [0, 1]]
N = 20
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

y = original_label.T 
X = X.T 

#############
dropout = .5 
eta = learning_rate = .01
d0 = 2 
d1 = 100 
d2 = 3 

def softmax(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

## One-hot coding
from scipy import sparse 
def convert_labels(y, C = 3):
    Y = sparse.coo_matrix((np.ones_like(y), 
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y 

def relu(y):
	return np.maximum(0, y)

def cost(Y, Yhat):
	# import pdb; pdb.set_trace()  # breakpoint 175213fd //
	return -np.sum(Y*np.log(Yhat))/Y.shape[1]

def dL(Y, Yhat):
	return (Yhat - Y)

def forward(x, W1,  b1, W2, b2, dropout, training = False):
	z1 = W1.T.dot(x) + b1 
	a1 = relu(z1)

	#dropout 
	if training: 
		# import pdb; pdb.set_trace()  # breakpoint 26f53f89 //
		mask = np.random.binomial(1, dropout, size = a1.shape)
	else:
		mask = dropout 
	a1 *= mask 

	z2 = W2.T.dot(a1) + b2 
	yhat = softmax(z2)

	return z1, a1, yhat, mask 

def backprop(x, z1, a1, yhat, mask, y, W1, W2, b1, b2):
	e2 = dL(y, yhat)
	dW2 = np.dot(a1, e2.T)
	db2 = e2 
	e1 = np.dot(W2, e2) 
	e1[z1 <=0] = 0 
	#dropout 
	# import pdb; pdb.set_trace()  # breakpoint e4e435b8 //
	e1 *= mask 
	dW1 = np.dot(x, e1.T)
	db1 = e1 

	# update 
	W1 += -eta*dW1 
	b1 += -eta*db1 
	W2 += -eta*dW2
	b2 += -eta*db2 
	return W1, b1, W2, b2

def get_sample(X, Y):
    for x, y in zip(X, Y):
        yield x[:, None], y[None,:] # makes sure the inputs are 2d row vectors

n_in = d0 
n_out = d2 

# import pdb; pdb.set_trace()  # breakpoint 5fb540b0 //
I = [np.random.randint(n_in, size=(n_in // 2 + 1)) for i in range(n_out)]

# initialize parameters randomely 
W1 = 0.01*np.random.randn(d0, d1)
b1 = np.zeros((d1, 1))
W2 = 0.01*np.random.randn(d1, d2)
b2 = np.zeros((d2, 1))
n_epoches = 1000
Y = convert_labels(y)
for epoch in range(n_epoches):

	# Training 
	# for x, y in get_sample(X, y):
	# i = np.random.randint(0, X.shape[1])
	idx = np.random.permutation(X.shape[1])

	for ii in xrange(X.shape[1]):
		i = idx[ii]
		x = X[:,i].reshape(2, 1)
		y = Y[:,i].reshape(3, 1)
		z1, a1, yhat, mask = forward(x, W1,  b1, W2, b2, dropout, training = True)
		W1, b1, W2, b2 = backprop(x, z1, a1, yhat, mask, y, W1, W2, b1, b2)

	# print(cost(y, yhat))


Z1 = np.dot(W1.T, X) + b1 
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
predicted_class = np.argmax(Z2, axis=0)
acc = (100*np.mean(predicted_class == y))
print('training accuracy: %.2f %%' % acc)