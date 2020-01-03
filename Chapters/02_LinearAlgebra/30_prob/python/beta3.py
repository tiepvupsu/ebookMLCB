
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

a, b = 12, 4
x = np.linspace(beta.ppf(0.00, a, b), beta.ppf(1, a, b), 100)
ax.plot(x, beta.pdf(x, a, b),'k-.', lw=2, alpha=1, label='$(12, 4)$')

a, b = 6, 2
x = np.linspace(beta.ppf(0.0, a, b), beta.ppf(1, a, b), 100)
ax.plot(x, beta.pdf(x, a, b),'k-', lw=2, alpha=1, label='$(6, 2)$')


a, b = 3, 1
x = np.linspace(beta.ppf(0.0, a, b), beta.ppf(0.99, a, b), 100)
ax.plot(x, beta.pdf(x, a, b),'k--', lw=2, alpha=1, label='$(3, 1)$')

a, b = 1.5, .5
x = np.linspace(beta.ppf(0.00, a, b), beta.ppf(.9, a, b), 1000)
ax.plot(x, beta.pdf(x, a, b),'k:', lw=2, alpha=1, label='$(1.5, 0.5)$')
ax.set_yticks([])
plt.xlim([0,1])
plt.ylim([0,6])
ax.legend(loc='upper right', fontsize = 14)
plt.xlabel('$\lambda$', fontsize = 15)
plt.ylabel('$p(\lambda)$', fontsize = 15)
with PdfPages('beta3.pdf') as pdf:
    pdf.savefig(bbox_inches='tight')
    
plt.show()