
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.stats import beta


fig, ax = plt.subplots(1, 1, figsize=(3.5, 4.5))
plt.tick_params(axis='both', which='major', labelsize=14)

a, b = .5, 1.5
x = np.linspace(beta.ppf(0.00, a, b), beta.ppf(1, a, b), 100)
#     ax.plot(x, beta.pdf(x, a, b),'k.', lw=2, alpha=1, label='beta pdf')

a, b = 4, 12
x = np.linspace(beta.ppf(0., a, b), beta.ppf(1, a, b), 100)
ax.plot(x, beta.pdf(x, a, b),'k:', lw=2, alpha=1, 
        label='$(4, 12)$')


a, b = 1, 3
x = np.linspace(beta.ppf(0.05, a, b), beta.ppf(1, a, b), 100)
ax.plot(x, beta.pdf(x, a, b),'k--', lw=2, alpha=1, label='$(1, 3)$')

a, b = 2, 6
x = np.linspace(beta.ppf(0.00, a, b), beta.ppf(1, a, b), 1000)
ax.plot(x, beta.pdf(x, a, b),'k-', lw=2, alpha=1, label='$(2, 6)$')

a, b = 0.25, .75
x = np.linspace(beta.ppf(0.3, a, b), beta.ppf(.99, a, b), 200)
ax.plot(x, beta.pdf(x, a, b),'k-.', lw=2, alpha=1, label='$(0.25, 0.75)$')

#     plt.text(0.2, .12, r'(0.25, 0.75)', fontsize=12, color = 'black')
#     plt.text(0.2, 1.94, r'(1, 3)', fontsize=12, color = 'black')
#     plt.text(0.3, 3.4, r'(4, 12)', fontsize=12, color = 'black')
# #     plt.text(0.2, 0.92, r'(.5, 1.5)', fontsize=12, color = 'black')
#     plt.text(0.35, 2.84, r'(2, 6)', fontsize=12, color = 'black')

plt.xlim([0,1])
plt.ylim([0,5])
ax.set_yticks([])

cur_axes = plt.gca()
#     cur_axes.axes.get_yaxis().set_ticks([0, 1.5])

plt.xlabel('$\lambda$', fontsize = 15)
plt.ylabel('$p(\lambda)$', fontsize = 15)
ax.legend(loc='upper right', fontsize = 14)
with PdfPages('beta2.pdf') as pdf:
    pdf.savefig(bbox_inches='tight')
plt.show() 