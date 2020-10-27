from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import time

#start = time.time()
#x, y = make_blobs(n_samples = 2 * 10**6, n_features = 50,
#                  centers = 10, cluster_std = 0.2,
#                  center_box = (-10.0, 10.0), random_state = 777)
#np.savetxt("kmeans_data.csv", x, fmt = "%f", delimiter = ",")
#print('{} sec'.format(time.time()-start))


import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.cluster import KMeans
# Set config to prevent long finiteness check of input data
skl.set_config(assume_finite = True)
# Fast CSV reading with pandas
start = time.time()
print('Data Loading Start...')
train_data = pd.read_csv("kmeans_data.csv", dtype = np.float32)
print('Data Load Time: {} sec'.format(time.time()-start))

alg = KMeans(n_clusters = 10, init = "random", tol = 0,
                         algorithm = "full", max_iter = 50)
start = time.time()
print('Training Start...')
alg.fit(train_data)
print('Training Time: {} sec'.format(time.time()-start))


