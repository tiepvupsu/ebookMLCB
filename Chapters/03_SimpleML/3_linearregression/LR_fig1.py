# -*- coding: utf8 -*-
from __future__ import print_function, unicode_literals
import matplotlib.font_manager as fm
set([f.name for f in fm.fontManager.ttflist])
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams['font.family'] = 'Dejavu Sans'


# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T # each row is a point 
# weight (kg)
y = np.array([ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# Building Xbar 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1) # each point is one row 
# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
# weights
w_0 = w[0]
w_1 = w[1]

x0 = np.linspace(145, 185, 2, endpoint=True)
y0 = w_0 + w_1*x0
plt.figure(figsize=(5,4))
# Drawing the fitting line 
plt.plot(X, y, 'o', color = 'white', mec = 'k', markersize = 8, label = u"dữ liệu huấn luyện")     # data 
plt.plot(x0, y0, color = 'k', linewidth = 2, label = u"mô hình tìm được")           # the fitting line
plt.axis([140, 190, 45, 75]) # xmin, xmax, ymin, ymax 
plt.xlabel(u'chiều cao (cm)', fontsize = 14)
plt.ylabel(u'cân nặng (kg)', fontsize = 14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(loc = "best", fontsize = 14) 
# with PdfPages('lr_ex.pdf') as pdf:
    # pdf.savefig(bbox_inches='tight')
# plt.savefig('lr_ex.png', bbox_inches='tight', dpi = 600)
plt.show()