# Tiny example of 3-layer nerual network with dropout in 2nd hidden layer
# Output layer is linear with L2 cost (regression model)
# Hidden layer activation is tanh

import numpy as np

n_epochs = 100
n_samples = 100
n_in = 10
n_hidden = 5
n_out = 4

dropout = 0.5 # 1.0 = no dropout
learning_rate = 0.01

def dtanh(y):
    return 1 - y**2

def C(y, t):
    # Cost function. y - model output; t - expected output/target
    return 0.5 * np.sum((t - y)**2) # 0.5 makes derivative nicer

def dC(y, t):
    return y - t

def forward(x, W1, W2, W3, dropout, training=False):
    z1 = np.dot(x, W1)
    y1 = np.tanh(z1)

    z2 = np.dot(y1, W2)
    y2 = np.tanh(z2)

    # Dropout in layer 2
    if training:
        m2 = np.random.binomial(1, dropout, size=z2.shape)
    else:
        m2 = dropout
    y2 *= m2

    z3 = np.dot(y2, W3)
    y3 = z3 # linear output

    return y1, y2, y3, m2

def backward(x, y1, y2, y3, m2, t, W1, W2, W3):
    dC_dz3 = dC(y3, t)
    dC_dW3 = np.dot(y2.T, dC_dz3)
    dC_dy2 = np.dot(dC_dz3, W3.T)

    dC_dz2 = dC_dy2 * dtanh(y2) * m2
    dC_dW2 = np.dot(y1.T, dC_dz2)
    dC_dy1 = np.dot(dC_dz2, W2.T)

    dC_dz1 = dC_dy1 * dtanh(y1)
    dC_dW1 = np.dot(x.T, dC_dz1)

    return dC_dW1, dC_dW2, dC_dW3

def update(W1, W2, W3, dC_dW1, dC_dW2, dC_dW3, learning_rate):
    # Gradient descent update
    W1 = W1 - learning_rate * dC_dW1
    W2 = W2 - learning_rate * dC_dW2
    W3 = W3 - learning_rate * dC_dW3

    return W1, W2, W3

def check_gradients(W1, W2, W3, dropout):
    # Numerically checks if our gradient computations are correct
    
    tiny = 1e-4

    x = np.random.uniform(size=(1, n_in))
    t = np.random.uniform(size=(1, n_out))

    W = [W1, W2, W3]

    for i in range(3):
        for j in range(W[i].shape[0]):
            for k in range(W[i].shape[1]):

                np.random.seed(1)
                y1, y2, y3, m2 = forward(x, W1, W2, W3, dropout, training=True)
                dW = backward(x, y1, y2, y3, m2, t, W1, W2, W3)

                gradient1 = dW[i][j,k]

                np.random.seed(1) # We wan't the same dropout mask to be generated
                W[i][j,k] -= tiny
                y1, y2, y3, m2 = forward(x, W1, W2, W3, dropout, training=True)
                cost1 = C(y3, t)

                np.random.seed(1)
                W[i][j,k] += 2*tiny
                y1, y2, y3, m2 = forward(x, W1, W2, W3, dropout, training=True)
                cost2 = C(y3, t)

                W[i][j,k] -= tiny # back to normal

                gradient2 = (cost2 - cost1) / (2*tiny)

                assert np.isclose(gradient1, gradient2), "%s != %s" % (gradient1, gradient2)

    print "Gradients OK"

def get_sample(X, Y):
    for x, y in zip(X, Y):
        yield x[None,:], y[None,:] # makes sure the inputs are 2d row vectors

W1 = np.random.uniform(low=-0.1, high=0.1, size=(n_in, n_hidden))
W2 = np.random.uniform(low=-0.1, high=0.1, size=(n_hidden, n_hidden))
W3 = np.random.uniform(low=-0.1, high=0.1, size=(n_hidden, n_out))

check_gradients(W1, W2, W3, dropout)

# Target is to learn some randomly generated function of the inputs
# (each output is a sum of a random subset of intputs)
# I - gives the indices of X elements to sum
I = [np.random.randint(n_in, size=(n_in / 2 + 1)) for i in range(n_out)]

X_train = np.random.uniform(size=(n_samples, n_in)) # Generates random samples
Y_train = np.hstack(X_train[:,idxs].sum(axis=1, keepdims=True) for idxs in I)

X_validation = np.random.uniform(size=(n_samples, n_in)) # Generates random samples
Y_validation = np.hstack(X_validation[:,idxs].sum(axis=1, keepdims=True) for idxs in I)

best_cost = np.inf

for epoch in range(n_epochs):

    # Training
    for x, t in get_sample(X_train, Y_train):
        y1, y2, y3, m2 = forward(x, W1, W2, W3, dropout, training=True)
        dC_dW1, dC_dW2, dC_dW3 = backward(x, y1, y2, y3, m2, t, W1, W2, W3)
        W1, W2, W3 = update(W1, W2, W3, dC_dW1, dC_dW2, dC_dW3, learning_rate)

    # Validation
    cost = 0.
    for x, t in get_sample(X_validation, Y_validation):
        _, _, y3, _ = forward(x, W1, W2, W3, dropout, training=False)
        cost += C(y3, t)
    print "Epoch: %d; Cost: %.3f" % (epoch+1, cost)

    # if cost < best_cost:
    #     best_cost = cost
    # else:
    #     break

print "Finished!"