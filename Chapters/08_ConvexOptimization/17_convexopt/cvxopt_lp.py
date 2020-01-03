from cvxopt import matrix, solvers
c = matrix([-5., -3.])
G = matrix([[1., 2., 1., -1., 0.], [1., 1., 4., 0., -1.]])
h = matrix([10., 16., 32., 0., 0])
solvers.options['show_progress'] = False
sol = solvers.lp(c, G, h)

print('Solution:')
print(sol['x'])

