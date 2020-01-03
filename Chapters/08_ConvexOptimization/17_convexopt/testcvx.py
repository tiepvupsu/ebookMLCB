from cvxopt import matrix, solvers
A = matrix([[-1.0, -1.0, 0., 1.0], [1., -1., -1., -1.]])
b = matrix([1.0, -2.0, 0., 4.])
c = matrix([[2.],[ 1.]])
sol = solvers.lp(c, A, b)
print(sol['x'])

import cvxopt