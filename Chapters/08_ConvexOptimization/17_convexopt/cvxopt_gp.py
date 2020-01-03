from cvxopt import matrix, solvers
from math import log, exp# gp
from numpy import array
import numpy as np 

K = [4]
F = matrix([[-1., 1., 1., 0.],
			[-1., 1., 0., 1.],
			[-1., 0., 1., 1.]])
g = matrix([log(40.), log(2.), log(2.), log(2.)])
solvers.options['show_progress'] = False
sol = solvers.gp(K, F, g)

print('Solution:')
print(np.exp(np.array(sol['x'])))

print('\nchecking sol^5')
print(np.exp(np.array(sol['x']))**5)
