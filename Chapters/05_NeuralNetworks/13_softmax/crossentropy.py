# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


x0 = np.linspace(0.001, 0.999, 1000)




def ce(p, q):
	return -(p*np.log(q) + (1-p)*np.log(1 - q))

def dist(p, q):
	return (p - q)**2 

fig = plt.figure(figsize=(4,4))
# fig = plt.figure(num=None, figsize=(4, 4), dpi=300)
# fig = plt.gcf()
# fig.set_size_inches(7, 10.5)

# fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')



# fig.suptitle('Example', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
# y0 = -(.5*np.log(x0) + .5*np.log(1 - x0))
p = .5 
y0 = ce(p, x0)
plt.plot(x0, y0, 'k-')
ax.text(0.1, 1.35, r'$-(0.5\log(q) + 0.5\log(1 - q))$', fontsize=15, color = 'black')

z0 = (p - x0)**2
plt.plot(x0, z0, 'k--')

plt.axis([0, 1, -.5, 6])
ax.text(0.2, .3, r'$(q - 0.5)^2$', fontsize=15, color = 'black')
ax.set_title("$p = 0.5$")
plt.plot(p, ce(p, p), 'kx', markersize= 7)
plt.plot(p, dist(p, p), 'kx', markersize= 7)

# plt.savefig('crossentropy1.png', bbox_inches='tight', dpi = 300)
plt.tick_params(axis='both', which='major', labelsize=14)
filename = 'crossentropy1.pdf'
with PdfPages(filename) as pdf:
    pdf.savefig(bbox_inches='tight')

# plt.show()
# #######################
plt.cla()
p = .1
y0 = ce(p, x0)
plt.plot(x0, y0, 'k-')
ax.text(0.07, 2, r'$-(0.1\log(q) + 0.9\log(1 - q))$', fontsize=15, color = 'black')

z0 = (p - x0)**2
plt.plot(x0, z0, 'k--')

plt.axis([0, 1, -.5, 6])
ax.text(0.6, .5, r'$(q - 0.1)^2$', fontsize=15, color = 'black')
ax.set_title("$p = 0.1$")
plt.plot(p, ce(p, p), 'kx', markersize= 7)
plt.plot(p, dist(p, p), 'kx', markersize= 7)

# plt.savefig('crossentropy2.png', bbox_inches='tight', dpi = 300)
plt.tick_params(axis='both', which='major', labelsize=14)
filename = 'crossentropy2.pdf'
with PdfPages(filename) as pdf:
    pdf.savefig(bbox_inches='tight')

# plt.show()

plt.cla() 
p = .8
y0 = ce(p, x0)
plt.plot(x0, y0, 'k-')
ax.text(0.15, 1.8, r'$-(0.8\log(q) + 0.2\log(1 - q))$', fontsize=15, color = 'black')

z0 = (p - x0)**2
plt.plot(x0, z0, 'k--')

plt.axis([0, 1, -.5, 6])
ax.text(0.1, .5, r'$(q - 0.8)^2$', fontsize=15, color = 'black')
ax.set_title("$p = 0.8$")
plt.plot(p, ce(p, p), 'kx', markersize= 7)
plt.plot(p, dist(p, p), 'kx', markersize= 7)
# plt.savefig('crossentropy3.png', bbox_inches='tight', dpi = 300)
plt.tick_params(axis='both', which='major', labelsize=14)
filename = 'crossentropy3.pdf'
with PdfPages(filename) as pdf:
    pdf.savefig(bbox_inches='tight')

plt.show()

