# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.mlab as mlab
np.random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)

# Building Xbar 
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_exact = np.dot(np.linalg.pinv(A), b)

###################

N = X.shape[0]
a1 = np.linalg.norm(y, 2)**2/N
b1 = 2*np.sum(X)/N
c1 = np.linalg.norm(X, 2)**2/N
d1 = -2*np.sum(y)/N 
e1 = -2*X.T.dot(y)/N



matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
xg = np.arange(1.5, 6.0, delta)
yg = np.arange(0.5, 4.5, delta)
Xg, Yg = np.meshgrid(xg, yg)
# Z = np.linalg.norm(Xg*Xbar[:, 0] + Yg*Xbar[:, 1] - y)**2
Z = a1 + Xg**2 +b1*Xg*Yg + c1*Yg**2 + d1*Xg + e1*Yg


# plt.figure()
# # CS = plt.contour(Xg, Yg, Z, [0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4])
# CS = plt.contour(Xg, Yg, Z, 50)
# manual_locations = [(4.5, 3.5), (4.2, 3), (4.3, 3.3), (3.7, 3.1)]
# plt.clabel(CS, inline=.1, fontsize=10, manual=manual_locations)
# plt.title('labels at selected locations')


# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(Xg, Yg, Z, cmap='jet')




def cost(w):
	return .5/Xbar.shape[0]*np.linalg.norm(y - Xbar.dot(w), 2)**2;

def grad(w):
	return 1/Xbar.shape[0] * Xbar.T.dot(Xbar.dot(w) - y)


def numerical_grad(w, cost):
	eps = 1e-4
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


def myGD(w_init, grad, eta):
	w = [w_init]
	for it in range(200):
		w_new = w[-1] - eta*grad(w[-1])

		# if np.linalg.norm(w_new - w[-1])/len(w_new) < 1e-3:
		if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
			break 
		w.append(w_new)
		# print('iter %d: ' % it, w[-1].T)
	return (w, it) 

# w_init = np.random.randn(2, 1)
w_init = np.array([[2], [1]])
(w1, it1) = myGD(w_init, grad, 0.05)
(w2, it2) = myGD(w_init, grad, 0.5)
(w3, it3) = myGD(w_init, grad, 1)

print(it1, it2, it3)
################################################################
############ Visualization############################

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 

# fig, ax = plt.subplots()
# # fig = plt.figure()
# plt.cla()

# x0 = np.linspace(0, 1, 2, endpoint=True)

# def update(ii):
# 	if ii == 0:
# 		plt.cla()
# 		plt.plot(X, y, 'b.')
# 		plt.axis([0, 1, 0, 15])
# 		y0 = w_exact[0][0] + w_exact[1][0]*x0
# 		plt.plot(x0, y0, color = 'green', linewidth = 2)

# 	y0 = w1[ii][0] + w1[ii][1]*x0
# 	animlist = plt.plot(x0, y0, 'r', alpha = 0.5 + .5* ii/it1) 
# 	ax.set_xlabel('1')
# 	return animlist, ax


# anim1 = FuncAnimation(fig, update, frames=np.arange(0, it1), interval=200)
# anim1.save('t1.gif', dpi=80, writer='imagemagick')

############ contour################

fig, ax = plt.subplots()
# fig = plt.figure()
plt.cla()

# x0 = np.linspace(0, 1, 2, endpoint=True)
plt.axis([1.5, 6, 0.5, 4.5])
def update2(ii):
	if ii == 0:
		plt.cla()
		CS = plt.contour(Xg, Yg, Z, 100)
		manual_locations = [(4.5, 3.5), (4.2, 3), (4.3, 3.3), (3.7, 3.1)]
		animlist = plt.clabel(CS, inline=.1, fontsize=10, manual=manual_locations)
		animlist = plt.title('labels at selected locations')
		# y0 = w_exact[0][0] + w_exact[1][0]*x0
		# plt.plot(x0, y0, color = 'green', linewidth = 2)
		plt.plot(w_exact[0], w_exact[1], 'go')

	# y0 = w1[ii][0] + w1[ii][1]*x0
	else:
		animlist = plt.plot([w3[ii-1][0], w3[ii][0]], [w3[ii-1][1], w3[ii][1]], 'r-')
	animlist = plt.plot(w3[ii][0], w3[ii][1], 'ro') 
	ax.set_xlabel('1')
	return animlist, ax


anim1 = FuncAnimation(fig, update2, frames=np.arange(0, it3), interval=200)
# anim1.save('t2.gif', dpi=80, writer='imagemagick')
# anim1.save('t2.gif', dpi=80, writer='ffmpeg', extra_args=['-vcodec', 'libxvid'])
plt.show()
# fig, ax1 = plt.subplots()
# fig2 = plt.figure()


# plt.subplot(131)
# plt.cla()

# x0 = np.linspace(0, 2, 2, endpoint=True)

# def update2(ii):
# 	if ii == 0:
# 		plt.cla()
# 		plt.plot(X, y, 'b.')
# 		plt.axis([0, 2, 0, 15])
# 		y0 = w_exact[0][0] + w_exact[1][0]*x0
# 		plt.plot(x0, y0, color = 'green', linewidth = 2)

# 	y0 = w2[ii][0] + w2[ii][1]*x0
# 	animlist = plt.plot(x0, y0, 'r', alpha = 0.5 + .5* ii/it2) 
# 	ax1.set_xlabel('1')
# 	return animlist, ax1


# anim2 = FuncAnimation(fig2, update2, frames=np.arange(0, it2), interval=200)
# anim2.save('t2.gif', dpi=80, writer='imagemagick')

# plt.show()