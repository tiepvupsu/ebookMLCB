# # import numpy as np
# # from mpl_toolkits.mplot3d import Axes3D
# # import matplotlib.pyplot as plt
# # import random

# # def fun(x, y):
# #   return x**2 + y**2

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # x = y = np.arange(-3.0, 3.0, 0.05)
# # X, Y = np.meshgrid(x, y)
# # zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
# # Z = zs.reshape(X.shape)

# # ax.plot_surface(X, Y, Z, color = 'k')

# # ax.set_xlabel('X Label')
# # ax.set_ylabel('Y Label')
# # ax.set_zlabel('Z Label')

# # plt.show()

# '''
# ======================
# 3D surface (color map)
# ======================

# Demonstrates plotting a 3D surface colored with the coolwarm color map.
# The surface is made opaque by using antialiased=False.

# Also demonstrates using the LinearLocator and custom formatting for the
# z axis tick labels.
# '''

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import numpy as np
# from matplotlib.backends.backend_pdf import PdfPages

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# # Make data.
# X = np.arange(-2, 2, 0.025)
# Y = np.arange(-2, 2, 0.025)
# X, Y = np.meshgrid(X, Y)
# Z = X**2 + Y**2 


# surf = ax.plot_surface(X, Y, Z, cmap=cm.jet,
#                        linewidth=0, antialiased=False)

# # Customize the z axis.
# ax.set_zlim(0, 5)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()

# fn = 'norm2_surf.png'
# plt.savefig(fn, bbox_inches='tight', dpi = 600)

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np 

choose = 4

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)

X = np.arange(-6, 6, 0.025)
Y = np.arange(-6, 6, 0.025)
X, Y = np.meshgrid(X, Y)
if choose == 1:
	Z = X**2 + Y**2
elif choose == 2: 
	Z = np.abs(X) + np.abs(Y) 
elif choose == 3: 
	Z = X**2 - Y**2
elif choose == 4:
	Z = 0.1*(X**2 + Y**2 - 10*np.sin(np.sqrt(X**2 + Y**2)))

ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=1, cmap=cm.jet)
cset = ax.contour(X, Y, Z, zdir='z', offset=-5, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='x', offset=0, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='y', offset=0, cmap=cm.coolwarm)

ax.set_xlabel('$x$', fontsize = 15)
ax.set_xlim(-6, 6)
ax.set_ylabel('$y$', fontsize =15)
ax.set_ylim(-6, 6)
if choose == 1:
	ax.set_zlabel('$f(x, y) = x^2 + y^2$', fontsize = 20)
elif choose == 2:
	ax.set_zlabel('$f(x, y) = |x| + |y|$', fontsize = 20)
elif choose == 3:
	ax.set_zlabel('$f(x, y) = x^2 - y^2$', fontsize = 20)
elif choose == 4:
	# ax.set_zlabel('$f(x, y) = \\frac{1}{10}(x^2 + 2y^2 - 2\sin(xy)$', fontsize = 20)
	ax.set_zlabel('$f(x, y)$', fontsize = 20)
ax.set_title('$f(x, y) = \\frac{1}{10}(x^2 + y^2 - 10\sin(\sqrt{x^2 + y^2})$ (nonconvex)', fontsize = 20)
ax.set_zlim(-5, 6)


# plt.savefig('aa.png')
plt.show()
