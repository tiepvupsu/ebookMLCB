import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = np.exp(X[i].dot(W))
    scores /= np.sum(scores) 
    loss -= np.log(scores[y[i]])
    scores[y[i]] -= 1
    dW += X[i].T.reshape(-1, 1).dot(scores.T.reshape(1, -1))

  loss /= num_train 
  loss += .5*reg*np.sum(W*W)
  dW /= num_train 
  dW += reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  num_train = X.shape[0]
  Z = X.dot(W)
  e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
  S = e_Z / e_Z.sum(axis = 1, keepdims = True)

  # B = np.log(A) 
  C = S[np.arange(S.shape[0]), y]

  loss -= np.sum(np.log(C)) 
  loss /= num_train
  loss += .5*reg*np.sum(W*W)
  
  ####
  S[np.arange(S.shape[0]), y] -= 1 
  dW += X.T.dot(S)
  dW /= num_train
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

