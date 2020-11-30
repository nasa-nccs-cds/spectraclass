import cudf, cuml, cupy, cupyx, os
from cuml.neighbors import NearestNeighbors
from cuml.datasets import make_blobs
from cupyx.scipy.sparse.csr import csr_matrix

nverts = 100000
ndims = 16
nneighbors = 5

X, _ = make_blobs( n_samples=nverts, centers=5, n_features=ndims, random_state=42 )
cuX = cudf.DataFrame(X)
print( f"\nINPUT DATA:\n{cuX.head(8)}")

model = NearestNeighbors( n_neighbors=nneighbors, verbose=True )
model.fit( cuX )
sparse_graph: csr_matrix = model.kneighbors_graph( cuX, mode="distance" )

reducer = cuml.UMAP( n_components=3 )
embedding = reducer.fit_transform( cuX, knn_graph = sparse_graph )
print(f"Completed embedding, shape = {embedding.shape}")
os.system("nvidia-smi")

if ...:



















