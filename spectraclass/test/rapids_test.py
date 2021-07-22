import cudf, cuml, cupy, cupyx
from cuml.neighbors import NearestNeighbors

# Using cudf Dataframe here is not likely to help with performance
# However, it's a good opportunity to get familiar with the API
source_df: cudf.DataFrame = cudf.read_csv('/att/nobackup/tpmaxwel/data/fashion-mnist-csv/fashion_train.csv')
data = source_df.loc[ :, source_df.columns[:-1] ]
target = source_df[ source_df.columns[-1] ]
n_neighbors=5

# fit model
model = NearestNeighbors( n_neighbors=5 )
model.fit(data)

# get nearest neighbors
dist_mlarr, ind_mlarr = model.kneighbors( data, return_distance=True )

# create sparse matrix
distances =  cupy.ravel( cupy.fromDlpack( dist_mlarr.to_dlpack() ) )
indices =    cupy.ravel( cupy.fromDlpack( ind_mlarr.to_dlpack() ) )
print( f"Computed KNN graph, distances shape = {distances.shape}, indices shape = {indices.shape}, distances[0:5]= {distances[0:5]}, indices[0:5]= {indices[0:5]}")
n_samples = indices.shape[0]
n_nonzero = n_samples * n_neighbors
rowptr = cupy.arange(0, n_nonzero + 1, n_neighbors)
knn_graph = cupyx.scipy.sparse.csr_matrix( ( distances, indices, rowptr ), shape=(n_samples, n_samples) )

print( f"Completed KNN, graph shape = {knn_graph.shape}" )

reducer = cuml.UMAP(
    n_neighbors=10,
    n_components=3,
    n_epochs=500,
    min_dist=0.1,
    output_type="numpy"
)
embedding = reducer.fit_transform( data, knn_graph = knn_graph )
print(f"Completed embedding, shape = {embedding.shape}")

# df = embedding.to_pandas()
# df.columns = ["x", "y"]
# df['cid'] = pd.Series([str(x) for x in target.to_array()], dtype="category")
#
# cvs = ds.Canvas(plot_width=400, plot_height=400)
# agg = cvs.points(df, 'x', 'y', ds.count_cat('cid'))
# img = tf.shade(agg, color_key=color_key, how='eq_hist')
#
# utils.export_image(img, filename='fashion-mnist', background='black')
#
# image = plt.imread('fashion-mnist.png')
# fig, ax = plt.subplots(figsize=(12, 12))
# plt.imshow(image)
# plt.setp(ax, xticks=[], yticks=[])
# plt.title("Fashion MNIST data embedded\n"
#           "into two dimensions by UMAP\n"
#           "visualised with Datashader",
#           fontsize=12)
#
# plt.show()
#





