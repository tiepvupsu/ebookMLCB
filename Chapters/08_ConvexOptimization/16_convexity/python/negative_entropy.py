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
x = np.arange(0, 5.0, delta)
y = np.arange(0, 15.0, delta)
X, Y = np.meshgrid(x, y)
Z = X*np.log(X) + Y*np.log(Y)

fig, ax = plt.subplots()
ax.axis('off')

frame1 = plt.gca()


filename = 'NE2.pdf'
with PdfPages(filename) as pdf:
# with PdfPages('multipage_pdf.pdf') as pdf:
    plt.figure(figsize=(3, 3))
    # plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')

    # CS = plt.contour(X, Y, Z, np.arange(0.1, 3, .25), colors='k')
    CS = plt.contour(X, Y, Z, np.arange(0.1, 3, .25))
    plt.axis('equal')
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.title('$f(x, y) = x\log(x) + y \log(y)$')
    # plt.axes.get_xaxis().set_visible(False)
    # plt.axes.get_yaxis().set_visible(False)
    plt.axis('off')
    # 
    pdf.savefig(bbox_inches='tight') # saves the current figure into a pdf page
    plt.close()
