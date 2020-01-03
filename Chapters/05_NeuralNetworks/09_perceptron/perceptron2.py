# generate data
# list of points 
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)

# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)

def predict(w, x):    
    return np.sign(np.dot(w, x))

def has_converged(X, y, w):
    return np.array_equal(predict(w, X), y) #True if h(w, X) == y else False

# def perceptron(X, y, w_init):
#     w = [w_init]
#     N = X.shape[1]
#     mis_points = []
#     while True:
#         # mix data 
#         mix_id = np.random.permutation(N)
#         for i in range(N):
#             xi = X[:, mix_id[i]]
#             yi = y[0, mix_id[i]]
#             if h(w[-1], xi)[0] != yi:
#                 mis_points.append(mix_id[i])
#                 w_new = w[-1] + yi*xi 
#                 w.append(w_new)
                
#         if has_converged(X, y, w[-1]):
#             break
#     return (w, mis_points)

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    mis_points = []
    t = 1
    while True:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in range(N):
            # import pdb; pdb.set_trace()  # breakpoint 048b3c9a //
            xi = X[:, mix_id[i]]
            yi = y[0, mix_id[i]]
            if predict(w[-1], xi) != yi:
                print i
                w_new = w[-1] + yi*xi 
                w.append(w_new)
                break 
        t += 1
        print 
        if has_converged(X, y, w[-1]):
            break
    return w



d = X.shape[0]
w_init = np.random.randn(d)
w = perceptron(X, y, w_init)