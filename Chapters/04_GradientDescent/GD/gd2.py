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




def cost(w):
	x = w[0]
	y = w[1]
	return (x**2 + y - 7)**2 + (x - y + 1)**2 
	# return x**2 + y**2

def grad(w):
	x = w[0]
	y = w[1]
	g = np.zeros_like(w)
	g[0] = 2*(x**2 + y - 7)*2*x + 2*(x - y + 1)
	g[1] = 2*(x**2 + y - 7)     + 2*(y - x - 1)
	return g


def numerical_grad(w, cost):
	eps = 1e-6
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
	return True if np.linalg.norm(grad1 - grad2) < 1e-4 else False 

w = np.random.randn(2, 1)
# w_init = np.random.randn(2, 1)
print( 'Checking gradient...', check_grad(w, cost, grad))

def myGD(w_init, grad, eta):
	w = [w_init]
	for it in range(200):
		w_new = w[-1] - eta*grad(w[-1])
		# print(w_new)
		print( np.linalg.norm(grad(w_new)))
		if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
			break 
		w.append(w_new)
		# print('iter %d: ' % it, w[-1].T)
	return (w, it) 

w_init = np.array([[-3], [-2.5]])
w_init = np.random.randn(2, 1)
(w1, it1) = myGD(w_init, grad, 0.05)
print(w1[-1])
# (w2, it2) = myGD(w_init, grad, 1)
# (w3, it3) = myGD(w_init, grad, 2)

# print(it1, it2, it3)
# print(w1, it1)


delta = 0.025
x = np.arange(-6.0, 5.0, delta)
y = np.arange(-20.0, 15.0, delta)
X, Y = np.meshgrid(x, y)
Z = (X**2 + Y - 7)**2 + (X-Y + 1)**2

# plt.figure()
# CS = plt.contour(X, Y, Z, np.concatenate((np.arange(0.1, 50, 5), np.arange(60, 200, 10))))
# manual_locations = [(-4, 15), (-2, 0), ( 1, .25)]
# plt.clabel(CS, inline=.1, fontsize=10, manual=manual_locations)
# plt.title('labels at selected locations')


import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 

w_init = np.random.randn(2, 1)

def save_gif2(eta):
    (w, it) = myGD(w_init, grad, eta)
    print(it)
    fig, ax = plt.subplots(figsize=(4,4))    
    plt.cla()
    plt.axis([-6, 5, -20, 15])
#     x0 = np.linspace(0, 1, 2, endpoint=True)
    
    def update(ii):
        if ii == 0:
            plt.cla()
            # CS = plt.contour(Xg, Yg, Z, 100)
            CS = plt.contour(X, Y, Z, np.concatenate((np.arange(0.1, 50, 5), np.arange(60, 200, 10))))
            manual_locations = [(-4, 15), (-2, 0), ( 1, .25)]
            animlist = plt.clabel(CS, inline=.1, fontsize=10, manual=manual_locations)
#             animlist = plt.title('labels at selected locations')
            # plt.plot(w_exact[0], w_exact[1], 'go')
        else:
            animlist = plt.plot([w[ii-1][0], w[ii][0]], [w[ii-1][1], w[ii][1]], 'r-')
        animlist = plt.plot(w[ii][0], w[ii][1], 'ro') 
        xlabel = '$\eta =$ ' + str(eta) + '; iter = %d/%d' %(ii, it)
        xlabel += '; ||grad||_2 = %.3f' % np.linalg.norm(grad(w[ii]))
        ax.set_xlabel(xlabel)
        return animlist, ax
       
    anim1 = FuncAnimation(fig, update, frames=np.arange(0, it), interval=200)
    fn = 'img2_' + str(eta) + '.gif'
    # anim1.save(fn, dpi=100, writer='imagemagick')

# save_gif2(1)
save_gif2(.01)



plt.show()
