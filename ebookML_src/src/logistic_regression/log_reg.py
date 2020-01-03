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

class myLogisticRegression(object):
    """docstring for myLogisticRegression"""
    def __init__(self, lr = 0.1, tol = 1e-4, nepoches = 100, check_every = 20):
        super(myLogisticRegression, self).__init__()
        self.lr = lr
        self.tol = tol
        self.nepoches = nepoches
        self.w = None 
        self.check_every = check_every 

    def prob(self, X):
        return sigmoid(X.dot(self.w))

    def predict(self, X, threshold = 0.5):
        """
        predict output of each row of X
        X: a numpy array of shape
        threshold: a threshold between 0 and 1 
        """
        return np.where(self.prob(X) > threshold)[0]

    def loss(self, X, y):
        # if self.bias: X = bias_trick(X)
        z = self.prob(X)
        return -np.mean(y*np.log(z) + (1-y)*np.log(1-z))

    def fit(self, X, y):
        N = X.shape[0]
        d = X.shape[1]
        self.w = w_old = np.random.randn(X.shape[1])
        ep = 0 
        while ep < self.nepoches: 
            ep += 1
            mix_ids = np.random.permutation(N)
            for i in mix_ids:
                xi = X[i]
                yi = y[i]
                zi = sigmoid(xi.dot(self.w))
                self.w = self.w + self.lr*(yi - zi)*xi 
                print(self.loss(X, y))
            if np.linalg.norm(self.w - w_old)/d < self.tol:
                break 

np.random.seed(2)
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

X = bias_trick(X)

model = myLogisticRegression( lr = 0.1) 
# print(X.shape)
model.fit(X, y)
print(model.prob(X))



















