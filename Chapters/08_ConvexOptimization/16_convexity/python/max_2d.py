"""
This is a demo of creating a pdf file with several pages,
as well as adding metadata and annotations to pdf files.
"""

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.



delta = 0.025
x = np.arange(-6.0, 5.0, delta)
y = np.arange(-20.0, 15.0, delta)
X, Y = np.meshgrid(x, y)
# Z = (X**2 + Y - 7)**2 + (X-Y + 1)**2

# Z = -X*np.log(0.8) - Y*np.log(0.2)
Z = np.maximum(np.abs(X) + 2*np.abs(Y), 2*X**2 + Y**2 - X*Y)

filename = 'max_2d.pdf'
with PdfPages(filename) as pdf:
# with PdfPages('multipage_pdf.pdf') as pdf:
    plt.figure(figsize=(3, 3))
    # plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')

    # CS = plt.contour(X, Y, Z, np.arange(0.5, 15, 1), colors='k')
    CS = plt.contour(X, Y, Z, np.arange(0.5, 15, 1))
    plt.axis('equal')
    plt.xlim(-3, 5)
    plt.ylim(-3.5, 3.5)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.title('$f(x, y) = \max(2x^2 + y^2 -xy, |x| + 2|y|)$', fontsize = 12)
    plt.axis('off')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
