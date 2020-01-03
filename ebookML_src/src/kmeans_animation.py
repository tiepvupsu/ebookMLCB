import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import matplotlib.animation as animation
import random
np.random.seed(4)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
X0 = np.random.multivariate_normal(means[0], cov, 500)
X1 = np.random.multivariate_normal(means[1], cov, 500)
X2 = np.random.multivariate_normal(means[2], cov, 500)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

def kmeans_init_centers(X, k):
    # randomly pick k rows of X
    return X[np.random.choice(X.shape[0], k)]
    # return X[[1, 5, 10], :]


def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis = 1)
    

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    max_it = 100
    it = 0 
    while it < max_it:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

(centers, labels, it) = kmeans(X, K)
print centers[-1]
print labels[-1].shape


###########
from matplotlib.animation import FuncAnimation 
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(centers[-1])

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

fig, ax = plt.subplots()

def update(ii):
    label2 = 'iteration {0}: '.format(ii/2)
    if ii%2:
        label2 += ' update centers'
    else:
        label2 += ' assign points to clusters'

    i_c = (ii+1)/2 
    i_p = ii/2

    label = labels[i_p]
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    animlist = plt.cla()
    animlist = plt.axis('equal')
    animlist = plt.axis([-2, 12, -3, 12])

    kwargs = {"markersize": 5, "alpha": .8, "markeredgecolor": 'k'}
    animlist = plt.plot(X0[:, 0], X0[:, 1], 'b^', **kwargs)
    animlist = plt.plot(X1[:, 0], X1[:, 1], 'go', **kwargs)
    animlist = plt.plot(X2[:, 0], X2[:, 1], 'rs', **kwargs)

    # display centers and voronoi 
    i = i_c 

    kwargs = {"markersize": 15, "markeredgecolor": 'k'}
    animlist = plt.plot(centers[i][0, 0], centers[i][0, 1], 'y^', **kwargs)
    animlist = plt.plot(centers[i][1, 0], centers[i][1, 1], 'yo', **kwargs)
    animlist = plt.plot(centers[i][2, 0], centers[i][2, 1], 'ys', **kwargs)

    colors = ['b', 'g', 'r']
    ## vonoroi 
    points = centers[i]
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius = 1000)
    for i,region in enumerate(regions):
        polygon = vertices[region]
        animlist = plt.fill(*zip(*polygon), alpha=.2, color = colors[i])


    ax.set_xlabel(label2)
    return animlist, ax

anim = FuncAnimation(fig, update, frames=np.arange(0, 2*it), interval=1000)
# anim.save('kmeans.gif', dpi=200, writer='imagemagick')
plt.show()