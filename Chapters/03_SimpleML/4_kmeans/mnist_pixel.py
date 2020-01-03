# %reset
import numpy as np 
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import normalize

from display_network import *

mndata = MNIST('../MNIST/')

is_train_set = False 
# is_train_set = True 
if is_train_set:
	mndata.load_training()
	X = mndata.train_images
else:
	mndata.load_testing()
	X = mndata.test_images
# X = np.asarray(X)
# X = normalize(X)

x0 = np.asarray(X[0], dtype = np.uint8)
x1 = np.reshape(x0, [28, 28])



# print x1


from PIL import Image
import numpy as np

w, h = 512, 512
# data = np.zeros((h, w, 3), dtype=np.uint8)

# data[256, 256] = [255, 0, 0]
img = Image.fromarray(x1)

img2 = img.resize((14, 14))
# img2.save('my.png')
# img2.show()
import scipy.misc
scipy.misc.imsave('my2.png', img2)

print np.asarray(img2)