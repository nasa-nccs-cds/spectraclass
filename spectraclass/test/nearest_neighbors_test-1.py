import cudf
import pandas as pd
import numpy as np
from cuml.datasets import make_blobs
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from sklearn.neighbors import NearestNeighbors as skNearestNeighbors

n_samples = 222232
n_features = 16
print_rows = 25

n_query = 10000
n_neighbors = 4
random_state = 0

device_data, _ = make_blobs(n_samples=n_samples,  n_features=n_features,  centers=5, random_state=random_state)
device_data = cudf.DataFrame(device_data)
host_data: pd.DataFrame = device_data.to_pandas()
print( f"Host dataframe: ndim = {host_data.ndim}, size = {host_data.size}, col shape = {host_data.columns.shape}, shape = {host_data.index.shape}, np-shape = {host_data.values.shape}" )

print( f"\n INPUT DATA:\n {device_data.head(25)}")

# Scikit-learn Model

knn_sk = skNearestNeighbors(algorithm="brute", n_jobs=-1)
knn_sk.fit(host_data)
D_sk, I_sk = knn_sk.kneighbors(host_data[:n_query], n_neighbors)

# cuML Model

knn_cuml = cuNearestNeighbors()
knn_cuml.fit(device_data)
D_cuml, I_cuml = knn_cuml.kneighbors(device_data[:n_query], n_neighbors)

print( f"\n D_cuml:\n {D_cuml.head(print_rows)}")
print( f"\n I_cuml:\n {I_cuml.head(print_rows)}")

print( f"\n D_sk:\n {D_sk[0:print_rows]}")
print( f"\n I_sk:\n {I_sk[0:print_rows]}")

# # Compare Results
#
# passed = np.allclose(D_sk, D_cuml.as_gpu_matrix(), atol=1e-3)
# print('compare knn: cuml vs sklearn distances %s'%('equal'if passed else 'NOT equal'))
#
# sk_sorted = np.sort(I_sk, axis=1)
# cuml_sorted = np.sort(I_cuml.as_gpu_matrix(), axis=1)
#
# diff = sk_sorted - cuml_sorted
#
# passed = (len(diff[diff!=0]) / n_samples) < 1e-9
# print('compare knn: cuml vs sklearn indexes %s'%('equal'if passed else 'NOT equal'))
#
