# list of points 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10

X0 = np.array([.5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5])
y0 = np.zeros_like(X0)

X1 = np.array([1.75, 2.25, 2.75, 3.25, 4, 4.25, 4.5, 4.75, 5, 5.5])
y1 = np.ones_like(X1)

plt.plot(X0, y0, 'ro', markersize = 8)
plt.plot(X1, y1, 'bs', markersize = 8)

plt.axis([0, 6, -.5, 1.5])
cur_axes = plt.gca()
plt.xlabel('hours studying')
plt.ylabel('fail(0) / pass(1)')
cur_axes.axes.get_yaxis().set_ticks([0, 1])
plt.axes().set_aspect('equal')
plt.savefig('ex1.png', bbox_inches='tight', dpi = 300)
plt.show()
