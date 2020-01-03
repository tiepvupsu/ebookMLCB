# list of points 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

x = np.linspace(-5, 5, 1000)
y1 = (np.sign(x) +1)/2
y2 = 1./(1 + np.exp(-x))
y3 = np.exp(x) /(np.exp(x) + np.exp(-x))
l1, = plt.plot(x, y1, 'r', linewidth = 2, label = "hard threshold")
l2, = plt.plot(x, y2, 'b', linewidth = 2, label = "$f(s) = \\frac{1}{1 + e^{-s}}$")
l3, = plt.plot(x, y3, 'g', linewidth = 2, label = "$f(s) = \\frac{e^{s}}{e^{s} + e^{-s}}$")
l4, = plt.plot([-1, 1], [-1, 2], 'y', linewidth = 2, label = "linear")
plt.plot([-4, 0], [1, 1], 'k -- ')

# handles, labels = ax.get_legend_handles_labels()
plt.legend(handles = [l1, l2, l3, l4], fontsize = 10)

cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([0, 1])

plt.axis([-4, 6, -.5, 1.4])
plt.axes().set_aspect('equal')
plt.savefig('activation.png', bbox_inches='tight', dpi = 300)
plt.show()

plt.close("all")