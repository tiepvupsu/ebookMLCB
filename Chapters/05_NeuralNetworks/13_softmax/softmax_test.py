import numpy as np 
from sklearn import linear_model           # for logistic regression
from sklearn.metrics import accuracy_score # for evaluation
from scipy import misc                     # for loading image
from scipy import sparse
np.random.seed(1)                          # for fixing random values



# randomly generate data 
N = 1000 # number of training sample 
d = 784 # data dimension 
C = 10 # number of classes 

X = np.random.randn(d, N)
y = np.random.randint(0, C, (N,))

# print(y)

def convert_labels(y):
    """
    convert 1d label to a matrix label: each column of this matrix 
    coresponding to 1 element in y. In i-th column of Y, only one non-zeros
    element located in the y[i]-th position, and = 1 
    ex: y = [0, 2, 1, 0], and 3 classes, then return 

            [[1, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0]]
    """
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), 
                          shape = (np.amax(y) + 1, len(y))).toarray()
    return Y 

Y = convert_labels(y)
# print(convert_labels(y))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # pass  # TODO: Compute and return softmax(x)
    x2 = x - np.amax(x)
    return np.exp(x2)/ np.sum(np.exp(x2), axis = 0)


def cost(X, Y, W):
    Z = softmax(W.T.dot(X))
    return -np.sum(Y*np.log(Z))

W_init = np.random.randn(d, C)

def grad(X, Y, W):
    Z = softmax(W.T.dot(X))
    V = Z - Y
    return X.dot(V.T)
    
def numerical_grad(X, Y, W, cost):
    eps = 1e-4
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy()
            W_n = W.copy()
            W_p[i, j] += eps 
            W_n[i, j] -= eps
            g[i,j] = (cost(X, Y, W_p) - cost(X, Y, W_n))/(2*eps)

    return g 

# g1 = grad(X[:, 0].reshape(d, D1), Y[:, 0].reshape(C, 1), W_init)
# g2 = numerical_grad(X[:, 0].reshape(d, 1), Y[:, 0].reshape(C, 1), W_init, cost)
g1 = grad(X, Y, W_init)
g2 = numerical_grad(X, Y, W_init, cost)

print(np.linalg.norm(g1 - g2))
# print(g1)
# print('')
# print(g2)
