from __future__ import print_function 
import numpy as np 

def check_grad(fn, gr, X):
    # convert X to an 1d array, later we'll need only one for loop
    X_flat    = X.reshape(-1)
    shape_X   = X.shape                # original shape of X 
    num_grad  = np.zeros_like(X)       # numerical grad, shape = shape of X 
    grad_flat = np.zeros_like(X_flat)  # 1d version of grad
    eps       = 1e-6            # a small number, 1e-10 -> 1e-6 is often good
    numElems  = X_flat.shape[0] # number of elements in X 
    # calculate numerical gradient 
    for i in range(numElems):          # iterate over all elements of X 
        Xp_flat      = X_flat.copy()
        Xn_flat      = X_flat.copy()
        Xp_flat[i]  += eps
        Xn_flat[i]  -= eps
        Xp           = Xp_flat.reshape(shape_X) 
        Xn           = Xn_flat.reshape(shape_X)
        grad_flat[i] = (fn(Xp) - fn(Xn))/(2*eps)

    num_grad = grad_flat.reshape(shape_X) 
    
    diff = np.linalg.norm(num_grad - gr(X))
    print('Difference between two methods should be small:', diff)

# ==== check if grad(trace(A*X)) == A^T ====
m, n = 10, 20
A = np.random.rand(m, n)
X = np.random.rand(n, m)

def fn1(X):
    return np.trace(A.dot(X))

def gr1(X):
    return A.T 

check_grad(fn1, gr1, X)

# ==== check if grad(x^T*A*x) == (A + A^T)*x  ====
A = np.random.rand(m, m)
x = np.random.rand(m, 1)

def fn2(x):
    return x.T.dot(A).dot(x) 

def gr2(x): 
    return (A+A.T).dot(x)

check_grad(fn2, gr2, x)
