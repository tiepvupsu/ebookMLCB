# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np 

import matplotlib.pyplot as plt

x0 = np.linspace(0.001, 0.999, 1000)




def ce(p, q):
	return -(p*np.log(q) + (1-p)*np.log(1 - q))

def dist(p, q):
	return (p - q)**2 

fig = plt.figure(figsize=(4.5, 4.5))
ax = fig.add_subplot(111)
p = .5 
y0 = ce(p, x0)
plt.plot(x0, y0, 'b-')
ax.text(0.2, 1.1, r'$-(0.5\log(q) + 0.5\log(1 - q)$', fontsize=14, color = 'blue')

z0 = (p - x0)**2
plt.plot(x0, z0, 'r-')

plt.axis([0, 1, -.5, 6])
ax.text(0.2, .3, r'$(q - 0.5)^2$', fontsize=14, color = 'red')
ax.set_title("$p = 0.5$")
plt.plot(p, ce(p, p), 'go', markersize= 5)
plt.plot(p, dist(p, p), 'go', markersize= 5)
plt.xlabel('$q$')

plt.savefig('crossentropy1.png', bbox_inches='tight', dpi = 800)
#######################
# # y0 = -(.5*np.log(x0) + .5*np.log(1 - x0))
plt.cla()
p = .1
y0 = ce(p, x0)
plt.plot(x0, y0, 'b-')
ax.text(0.1, 1.3, r'$-(0.1\log(q) + 0.9\log(1 - q)$', fontsize=14, color = 'blue')

z0 = (p - x0)**2
plt.plot(x0, z0, 'r-')

plt.axis([0, 1, -.5, 6])
ax.text(0.5, .4, r'$(q - 0.1)^2$', fontsize=14, color = 'red')
ax.set_title("$p = 0.1$")
plt.plot(p, ce(p, p), 'go', markersize= 5)
plt.plot(p, dist(p, p), 'go', markersize= 5)
plt.xlabel('$q$')
plt.savefig('crossentropy2.png', bbox_inches='tight', dpi = 800)



# #######################
# ax = fig.add_subplot(133)
# y0 = -(.5*np.log(x0) + .5*np.log(1 - x0))
plt.cla()
p = .8
y0 = ce(p, x0)
plt.plot(x0, y0, 'b-')
ax.text(0.3, 1.2, r'$-(0.8\log(q) + 0.2\log(1 - q)$', fontsize=14, color = 'blue')

z0 = (p - x0)**2
plt.plot(x0, z0, 'r-')

plt.axis([0, 1, -.5, 6])
ax.text(0.2, .5, r'$(q - 0.8)^2$', fontsize=14, color = 'red')
ax.set_title("$p = 0.8$")
plt.plot(p, ce(p, p), 'go', markersize= 5)
plt.plot(p, dist(p, p), 'go', markersize= 5)
# plt.tight_layout()
plt.xlabel('$q$')
plt.savefig('crossentropy3.png', bbox_inches='tight', dpi = 800)
# plt.show()