
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

a, b = 2, 2
x = np.linspace(beta.ppf(0.00, a, b), beta.ppf(1, a, b), 100)
ax.plot(x, beta.pdf(x, a, b),'k-.', lw=2, alpha=1, label='$(2, 2)$')

a, b = 1, 1
x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
ax.plot(x, beta.pdf(x, a, b),'k-', lw=2, alpha=0.6)


a, b = .5, .5
x = np.linspace(beta.ppf(0.05, a, b), beta.ppf(0.95, a, b), 100)
ax.plot(x, beta.pdf(x, a, b),'k--', lw=2, alpha=1, label='$(0.5, 0.5)$')

a, b = 10, 10
x = np.linspace(beta.ppf(0.00, a, b), beta.ppf(1, a, b), 1000)
ax.plot(x, beta.pdf(x, a, b),'k-', lw=2, alpha=1, label='$(10, 10)$')

a, b = 0.1, .1
x = np.linspace(beta.ppf(0.3, a, b), beta.ppf(.7, a, b), 500)
ax.plot(x, beta.pdf(x, a, b),'k:', lw=2, alpha=1, label='$(0.1, 0.1)$')

# plt.text(0.4, -.1, r'(0.1, 0.1)', fontsize=12, color = 'black')
# plt.text(0.4, .3, r'(0.5, 0.5)', fontsize=12, color = 'black')
plt.text(0.4, 1.1, r'(1, 1)', fontsize=12, color = 'black')
# plt.text(0.4, 1.6, r'(2, 2)', fontsize=12, color = 'black')
# plt.text(0.4, 3.6, r'(10, 10)', fontsize=12, color = 'black')

ax.set_yticks([])
plt.xlim([0,1])

plt.xlabel('$\lambda$', fontsize = 15)
plt.ylabel('$p(\lambda)$', fontsize = 15)
ax.legend(loc='best', fontsize = 14)
with PdfPages('beta1.pdf') as pdf:
    pdf.savefig(bbox_inches='tight')
plt.show()