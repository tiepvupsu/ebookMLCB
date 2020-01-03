## Latex font in ipython
```python
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

from __future__ import print_function 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib.backends.backend_pdf import PdfPages
np.random.seed(18)
```

## pdf 
```python
filename = 
with PdfPages(filename) as pdf:
    pdf.savefig(bbox_inches='tight')
```

## tick fontsize 
```python
plt.tick_params(axis='both', which='major', labelsize=14)
```

# markercoloredge

# subplots 
```python
nrows = 4
ncols = 4
width = 4*ncols
height = 4*nrows
plt.close('all')
fig,axs=plt.subplots(nrows,ncols,figsize=(width,height)) 
for i, k in enumerate(ids):
    r = i//ncols 
    c = i%ncols 
    axs[r, c].plot(x0, y0, 'b')
```

# figuresize 
```python
plt.figure(figsize=(5.5,4))
```

# text
```python 
plt.text(0.6, 0.6, r'$\mathcal{A}\mathrm{sin}(2 \omega t)$',
         fontsize=20)
```