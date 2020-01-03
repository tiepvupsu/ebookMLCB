# -*- coding: utf8 -*-
from __future__ import print_function, unicode_literals
import matplotlib.font_manager as fm
from matplotlib import rc

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# plt.rcParams['font.family'] = 'Dejavu Sans'

rc('text', usetex=True)

set([f.name for f in fm.fontManager.ttflist])
#####
plt.scatter([1], [1])
plt.xlabel(u'vũ')
plt.ylabel('$y$')
plt.show()