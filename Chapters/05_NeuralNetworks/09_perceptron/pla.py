import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(16)

means = [[2, 2], [5, 3]]
cov = [[1, 0], [0, 1]]
N = 5
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 8, alpha = .8)
plt.plot(X1[:, 0], X1[:, 1], 'ro', markersize = 8, alpha = .8)
plt.axis('equal')
# plt.show()

X = np.concatenate((X0, X1), axis = 0)

original_label =y = np.asarray([-1]*N + [1]*N).T

# Building Xbar 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1).T 
X = Xbar 
# print(Xbar)
# print(y)

def h(w, x):    
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):
    print(h(w, X)[0])
    return np.array_equal(h(w, X)[0], y) #True if h(w, X) == y else False

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    mis_points = []
    while True:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(3, 1)
            yi = y[mix_id[i]]
            if h(w[-1], xi)[0] != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi*xi 

                w.append(w_new)
                
        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)

d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, mis_points) = perceptron(X, y, w_init)
print(w)
# print(w)
print( h(w[-1], Xbar))            
print(len(w))

# def draw_line(w):
#     w0, w1, w2 = w[0], w[1], w[2]
#     if w2 != 0:
#         x11, x12 = 0, 10
#         return plt.plot([x11, x12], [-(w1*x11 + w0)/w2, -(w1*x12 + w0)/w2])
#     else:
#         x10 = -w0/w1
#         return plt.plot([x10, x10], [0, 10])

#     plt.cla()
# draw_line([1, 2, 1])
# plt.show()
    

# ## GD example
# import matplotlib.animation as animation
# from matplotlib.animation import FuncAnimation 
# def viz_alg_1d_2(w):
#     it = len(w)    
       
#     fig, ax = plt.subplots(figsize=(4, 4))  
    
#     def update(i):
#         ani = plt.cla()
#         #points
#         ani = plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
#         ani = plt.plot(X1[:, 0], X1[:, 1], 'ro', markersize = 4, alpha = .8)
#         ani = plt.axis([0 , 6, 0, 6])
#         ani = draw_line(w[ i if i < it else it-1 ])
#         label = 'GD without Momemtum: iter %d/%d' %(i, it)
#         ax.set_xlabel(label)
#         return ani, ax 
        
#     anim = FuncAnimation(fig, update, frames=np.arange(0, it + 10), interval=500)
#     anim.save('haha.gif', dpi = 100, writer = 'imagemagick')
#     plt.show()
    
# # x = np.asarray(x)
# viz_alg_1d_2(w)
