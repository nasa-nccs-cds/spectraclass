import cudf, os
from cuml.neighbors import NearestNeighbors
from cuml.datasets import make_blobs

nverts = 25
X, _ = make_blobs( n_samples=nverts, centers=5, n_features=10, random_state=42 )

# build a cudf Dataframe
X_cudf = cudf.DataFrame(X)

# fit model
model = NearestNeighbors(n_neighbors=3)
model.fit(X)

# get 3 nearest neighbors
distances, indices = model.kneighbors(X_cudf)
os.system("nvidia-smi")

# print results
print( f"\nINDICES:\n{indices.head(20)}")
print(f"\nDISTANCES:\n{distances.head(20)}")