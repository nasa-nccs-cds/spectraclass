import cudf, cuml, cupy, cupyx
from cuml.neighbors import NearestNeighbors
from cuml.datasets import make_blobs
from cupyx.scipy.sparse.csr import csr_matrix

nverts = 100000
ndims = 16
nneighbors = 5

X, _ = make_blobs( n_samples=nverts, centers=5, n_features=ndims, random_state=42 )
X_cudf = cudf.DataFrame(X)
print( f"\nINPUT DATA:\n{X_cudf.head(10)}")

model = NearestNeighbors(n_neighbors=nneighbors)
model.fit(X)
cu_sparse_graph: csr_matrix = model.kneighbors_graph(X_cudf)
sparse_graph = cu_sparse_graph.get()

#offsets = cudf.Series(sparse_graph.indptr)
#indices = cudf.Series(sparse_graph.indices)
#offsets = sparse_graph.indptr
#indices = sparse_graph.indices

reducer = cuml.UMAP(
    n_neighbors=10,
    n_components=3,
    n_epochs=500,
    min_dist=0.1,
    output_type="numpy"
)

embedding = reducer.fit_transform( X_cudf, knn_graph = cu_sparse_graph )
print(f"Completed embedding, shape = {embedding.shape}")






