import cudf, os
from cuml.neighbors import NearestNeighbors
from cuml.datasets import make_blobs
from cupyx.scipy.sparse.csr import csr_matrix
nverts = 500000
ndims = 16
nneighbors = 5

X, _ = make_blobs( n_samples=nverts, centers=5, n_features=ndims, random_state=42 )

# build a cudf Dataframe
X_cudf = cudf.DataFrame(X)

print( f"\nINPUT DATA:\n{X_cudf.head(10)}")

# fit model
model = NearestNeighbors(n_neighbors=nneighbors)
model.fit(X)

# get 3 nearest neighbors
distances, indices = model.kneighbors(X_cudf)
os.system("nvidia-smi")

# print results
print( f"\nINDICES:\n{indices.head(10)}")
print(f"\nDISTANCES:\n{distances.head(10)}")
