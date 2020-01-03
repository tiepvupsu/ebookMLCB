
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
from matplotlib.backends.backend_pdf import PdfPages

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np 



fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(1)

X = np.arange(-2, 2, 0.025)
Y = np.arange(-2, 2, 0.025)
X, Y = np.meshgrid(X, Y)

Z = np.exp(-(X**2 + Y**2 - .5*X*Y))*1.5


ax.plot_surface(X, Y, Z, alpha=1, cmap=cm.gray)
cset = ax.contour(X, Y, Z, zdir='z', offset=-1, cmap=cm.gray)

ax.set_xlabel('$x$', fontsize = 15)
ax.set_xlim(-2, 2)
ax.set_ylabel('$y$', fontsize =15)
ax.set_ylim(-2, 2)

ax.set_zlabel('$p(x, y)$', fontsize = 15)

# ax.set_title('bivariate normal distribution', fontsize = 15)
ax.set_zlim(-1, 1)
ax.set_zticks([])
ax.set_xticks([-2,  0,  2])
ax.set_yticks([-2,  0,  2])
# plt.axis('equal')
# ax.auto_scale_xyz([0, 500], [0, 500], [0, 0.15])
# plt.savefig('aa.png')

plt.tick_params(axis='both', which='major', labelsize=14)

with PdfPages('bi_norm.pdf') as pdf:
    pdf.savefig(bbox_inches='tight')
# plt.savefig('aa.png')
# plt.savefig('bi_norm.p')
plt.show()



plt.show()

