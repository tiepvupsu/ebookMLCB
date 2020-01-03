# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np 

import matplotlib.pyplot as plt

alpha = .5
x = [5.5]
c_sin = 8
title = '$f(x) = x^2 + %dsin(x)$; ' %c_sin
title += '$x_0 =  %.2f$; ' %x[0]
title += r'$\alpha = %.2f$ ' % alpha 
file_name = 'gd_14.gif'

def grad(x):
    return 2*x+ c_sin*np.cos(x)

def cost(x):
    return x**2 + c_sin*np.sin(x)



for it in range(100):
	# print cost(x[-1])
	x_new = x[-1] - alpha*grad(x[-1])

	if np.linalg.norm(x_new - x[-1]) < 1e-3:
		break 
	x.append(x_new)

print( it )
x = np.asarray(x)
# print it, x_old
x0 = np.linspace(-4.5, 5.5, 1000)
y0 = cost(x0)

y = cost(x)
g = grad(x)ls
plt.plot(x0, y0)
plt.plot(x, y, 'ro', markersize=7)


fig, ax = plt.subplots()

def update(ii):
    label2 = 'iteration %d/%d: ' %(ii, it) + 'cost = %.2f' % y[ii] + ', grad = %.4f' %g[ii]

    animlist = plt.cla()
    # animlist = plt.axis('equal')
    animlist = plt.axis([-6, 6, -8, 30])

    animlist = plt.plot(x0, y0)
    
    # title += '$\alpha = $ %2f' % alpha
    # animlist = plt.title('$x_0 = $%f, $\alpha = $%f' % (x[0], alpha))
    animlist = plt.title(title)
    if ii == 0:
    	animlist = plt.plot(x[ii], y[ii], 'ro', markersize = 7)
    else:
    	animlist = plt.plot(x[ii-1], y[ii-1], 'ko', markersize = 7)
    	animlist = plt.plot([x[ii-1], x[ii]], [y[ii-1], y[ii]], 'k-')
    	animlist = plt.plot(x[ii], y[ii], 'ro', markersize = 7)

    ax.set_xlabel(label2)
    return animlist, ax
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 

anim = FuncAnimation(fig, update, frames=np.arange(0, it), interval=500)
anim.save(file_name, dpi=100, writer='imagemagick')
plt.show()

