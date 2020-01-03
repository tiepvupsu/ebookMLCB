from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Palatino']})

import matplotlib.font_manager as fm
set([f.name for f in fm.fontManager.ttflist])
rc('text', usetex=True)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
from matplotlib.backends.backend_pdf import PdfPages


mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu-5*variance,mu+5*variance, 200)
plt.plot(x,mlab.normpdf(x, mu, sigma), 'k')

mu = 0.5
variance = 0.5
sigma = math.sqrt(variance)
x = np.linspace(mu-6*variance,mu+6*variance, 200)
plt.plot(x,mlab.normpdf(x, mu, sigma), 'k')
plt.tick_params(axis='both', which='major', labelsize=14)
mu = -1
variance = 2
sigma = math.sqrt(variance)
x = np.linspace(mu-4*variance,mu+4*variance, 200)
plt.plot(x,mlab.normpdf(x, mu, sigma), 'k')
plt.xlabel('$x$', fontsize = 15)
plt.ylabel('$p(x)$', fontsize = 15)
plt.plot([-8, 8], [0, 0], 'k',)

plt.text(-5.75, .35, r'$\mu = 0, \sigma^2 = 1$', fontsize=15, color = 'black')
plt.text(1.25, .5, r'$\mu = .5, \sigma^2 = .5$', fontsize=15, color = 'black')
plt.text(-8.25, .15, r'$\mu = -1, \sigma^2 = 2$', fontsize=15, color = 'black')
# plt.title(u'phân phối chuẩn một chiều')
# plt.text(2, 6)
with PdfPages('uni_norm.pdf') as pdf:
    pdf.savefig(bbox_inches='tight')
# plt.savefig('uni_norm.png', bbox_inches = 'tight', dpi = 600)
plt.show()