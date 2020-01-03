print('Tiep')
c = [-1, 4]
A = [[-3, 1], [1, 2]]
b = [6, 4]
x0_bounds = (None, None)
x1_bounds = (-3, None) 
from scipy.optimize import linprog 
res = linprog(c, A_ub=A, b_ub=b, bounds=((0, None)), options={"disp": True})
print(res.x)