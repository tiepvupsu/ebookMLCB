# list of points 
import numpy as np 
import matplotlib.pyplot as plt
# from scipy.spatial.distance import cdist
from sklearn import datasets, linear_model
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10

X0 = np.array([.5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5])
y0 = np.zeros_like(X0)

X1 = np.array([1.75, 2.25, 2.75, 3.25, 4, 4.25, 4.5, 4.75, 9, 20])
y1 = np.ones_like(X1)

X = np.concatenate((X0, X1)).reshape(len(X0) + len(X1), 1)
y = np.concatenate((y0, y1)).reshape(len(X0) + len(X1), 1)

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

w = regr.coef_ 
print(w, w.shape)

xx = np.linspace(0, 21, 2)
yy = w[0][0] + w[0][1]*xx 

xx1 = (.5 -w[0][0])/w[0][1]
plt.plot(xx, yy, 'y-', linewidth = 2)
plt.plot(xx1, 0.5, 'gx', markersize=10)




plt.plot(X0, y0, 'ro', markersize = 8)
plt.plot(X1, y1, 'bs', markersize = 8)

plt.axis([0, 20.5, -.5, 1.5])
cur_axes = plt.gca()
plt.xlabel('hours studying')
plt.ylabel('fail(0) / pass(1)')
cur_axes.axes.get_yaxis().set_ticks([0, 1])
plt.axes().set_aspect('equal')
plt.savefig('ex1_lr.png', bbox_inches='tight', dpi = 300)
plt.show()
