import cudf, os
from cuml.neighbors import NearestNeighbors
from cuml.datasets import make_blobs
from cupyx.scipy.sparse.csr import csr_matrix
import scipy

nverts = 100000
ndims = 16
nneighbors = 5

X, _ = make_blobs( n_samples=nverts, centers=5, n_features=ndims, random_state=42 )

# build a cudf Dataframe
X_cudf = cudf.DataFrame(X)

print( f"\nINPUT DATA:\n{X_cudf.head(10)}")

# fit model
model = NearestNeighbors(n_neighbors=nneighbors)
model.fit(X)

# get nearest neighbors
cu_sparse_graph: csr_matrix = model.kneighbors_graph(X_cudf)
sparse_graph = cu_sparse_graph.get()
#offsets = cudf.Series(sparse_graph.indptr)
#gindices = cudf.Series(sparse_graph.gindices)

offsets = sparse_graph.indptr
indices = sparse_graph.gindices

#print( f"offsets:\n {offsets}" )
print( f"indices:\n {indices[0:100].reshape(20,5)}" )

# os.system("nvidia-smi")

# print results

