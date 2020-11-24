import cudf
from cuml.neighbors import NearestNeighbors
from cuml.datasets import make_blobs

X, _ = make_blobs(n_samples=25, centers=5,
                  n_features=10, random_state=42)
print(f"Input shape = {X.shape}")

# build a cudf Dataframe
X_cudf = cudf.DataFrame(X)

print(f"X_cudf head = {X_cudf.head(10)}")

# fit model
model = NearestNeighbors(n_neighbors=3)
model.fit(X)

# get 3 nearest neighbors
distances, indices = model.kneighbors(X_cudf)

# print results
print(indices)
print(distances)

