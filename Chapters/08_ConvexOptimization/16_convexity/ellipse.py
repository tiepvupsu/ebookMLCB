# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(4)

x = np.sqrt(2) 

# A = np.array([[x/2, -x], [x/2, x]])

# B = np.linalg.inv(A)
# print(B)
B = np.array([[x, x], [-x/2, x/2]])
A = B 
def pred(X, A):
    res = np.zeros((1, X.shape[1]))
    # print(res.shape)

    for i in xrange(X.shape[1]):
        xi = X[:, i].reshape(2, 1)
        res[0, i] = xi.T.dot(np.linalg.inv(A)).dot(xi) <= 1
    return res 
    
xm = np.arange(-3, 4, 0.025)
xlen = len(xm)
ym = np.arange(-4, 4, 0.025)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)

# print(np.ones((1, xx.size)).shape)
xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)

X0 = np.vstack((xx1, yy1))

print(X0.shape)
Z = pred(X0, A)

print(xx.shape)
Z = Z.reshape(xx.shape[0], xx.shape[1])
CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha = .1)
plt.axis('equal')

# kmeans_display(X.T, original_label.T)

# cur_axes = plt.gca()
# cur_axes.axes.get_xaxis().set_ticks([])
# cur_axes.axes.get_yaxis().set_ticks([])

# plt.title('$\lambda =$' + str(lam), fontsize = 20)
# fn = 'nnet_reg'+ str(lam) + '.png'
# plt.savefig(fn, bbox_inches='tight', dpi = 600)

plt.show()