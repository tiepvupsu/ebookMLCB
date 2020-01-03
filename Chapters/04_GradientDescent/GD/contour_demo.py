#!/usr/bin/env python
"""
Illustrate simple contour plotting, contours on an image with
a colorbar for the contours, and labelled contours.

See also contour_image.py.
"""
from __future__ import division, print_function, unicode_literals
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

# delta = 0.025
# x = np.arange(-6.0, 5.0, delta)
# y = np.arange(-20.0, 15.0, delta)
# X, Y = np.meshgrid(x, y)
# Z = (X**2 + Y - 7)**2 + (X-Y + 1)**2

# plt.figure()
# CS = plt.contour(X, Y, Z, np.concatenate((np.arange(0.1, 50, 5), np.arange(60, 200, 10))))
# manual_locations = [(-4, 15), (-2, 0), ( 1, .25)]
# plt.clabel(CS, inline=.1, fontsize=10, manual=manual_locations)
# plt.title('labels at selected locations')


def cost(w):
	x = w[0]
	y = w[1]
	return (x**2 + y - 7)**2 + (x - y + 1)**2 

def grad(w):
	x = w[0]
	y = w[1]
	g0 = 2*(x**2 + y - 7)*2*x + 2*(x - y + 1)
	g1 = 2*(x**2 + y - 7)     + 2*(y - x - 1)
	return np.array([[g0], [g1]])


def numerical_grad(w, cost):
	eps = 1e-10
	g = np.zeros_like(w)
	for i in range(len(w)):
		w_p = w.copy()
		w_n = w.copy()
		w_p[i] += eps 
		w_n[i] -= eps
		g[i] = (cost(w_p) - cost(w_n))/(2*eps)
	return g 

def check_grad(w, cost, grad):
	w = np.random.rand(w.shape[0], w.shape[1])
	grad1 = grad(w)
	grad2 = numerical_grad(w, cost)
	return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False 


print( 'Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))
print(grad(np.random.rand(2, 1)))
print(numerical_grad(np.random.rand(2, 1), cost))


plt.show()
