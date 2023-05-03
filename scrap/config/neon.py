c.ModeDataManager.cache_dir = "/explore/nobackup/projects/ilab/cache"
c.ModeDataManager.data_dir  = "/explore/nobackup/projects/ilab/data"
c.ModeDataManager.model_dims = 32
c.ModeDataManager.reduce_nepoch = 2
c.ModeDataManager.reduce_niter = 12
c.ModeDataManager.reduce_sparsity = 0.0
c.ModeDataManager.subsample_index = 1
c.ModeDataManager.reduce_method = "Autoencoder"
c.ModeDataManager.class_file = "NONE"
c.ModeDataManager.refresh_model = False
c.ModeDataManager.modelkey = "neon150.aec"
c.ModeDataManager.reduce_anom_focus = 0.15
c.ModeDataManager.reduce_focus_nepoch = 4
c.ModeDataManager.reduce_dropout = 0.0
c.ModeDataManager.reduce_focus_ratio = 10.0
c.ModeDataManager.reduce_learning_rate = 0.0001
c.ModeDataManager.reduce_nblocks = 1000
c.ModeDataManager.reduce_nimages = 100

c.DataManager.proc_type = "cpu"
c.DataManager.refresh_data = False
c.DataManager.labels_dset = "labels"
c.DataManager.preprocess = False

c.TileManager.block_index = (1, 5)
c.TileManager.block_size = 150
c.TileManager.mask_class = 0
c.TileManager.autoprocess = False
c.TileManager.reprocess = False

c.EmbeddingManager.alpha = 0.9
c.EmbeddingManager.init = "random"
c.EmbeddingManager.nepochs = 200
c.EmbeddingManager.target_weight = 0.5

c.ReductionManager.alpha = 0.9
c.ReductionManager.init = "random"
c.ReductionManager.loss = "mean_squared_error"
c.ReductionManager.ndim = 3
c.ReductionManager.nepochs = 200
c.ReductionManager.target_weight = 0.5

c.ActivationFlowManager.metric = "cosine"
c.ActivationFlowManager.nneighbors = 5

c.MapManager.init_band = 10
c.MapManager.lower_threshold = 0.0
c.MapManager.upper_threshold = 1.0

c.TextureManager.textures = [{'type': 'gabor', 'bands': [1, 3], 'nfeatures': 1}, {'type': 'glcm', 'bands': [1], 'features': ['homogeneity', 'energy']}]

c.ClusterManager.modelid = "kmeans"
c.ClusterManager.nclusters = 10
c.ClusterManager.random_state = 0

c.ClassificationManager.mid = "mlp"
c.ClassificationManager.nfeatures = 32
c.ClassificationManager.nepochs = 10

c.PointCloudManager.color_map = "gist_rainbow"

