# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{mathtools}

\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage[T1]{fontenc}
% \usepackage[utf8]{inputenc}
% \usepackage[vietnam]{babel}
% more packages here
'''

plt.imshow(np.random.randn(100, 100))
# plt.title(u'hữu'.encode('ascii').decode('ascii'))
plt.title('vũ hữu')
plt.xlabel('$x$')

plt.ylabel('$y$')
# plt.savefig('test.pdf')
plt.show()