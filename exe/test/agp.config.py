c.ModeDataManager.model_dims = 32
c.ModeDataManager.reduce_sparsity = 0.0
c.ModeDataManager.subsample_index = 1
c.ModeDataManager.class_file = "NONE"
c.DataManager.proc_type = "skl"
c.DataManager.refresh_data = False
c.DataManager.labels_dset = "labels"
c.TileManager.block_index = (1,7)
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
c.PointCloudManager.color_map = "gist_rainbow"
c.MapManager.init_band = 0
c.MapManager.lower_threshold = 0.0
c.MapManager.upper_threshold = 1.0
c.TextureManager.textures = [{'type': 'gabor', 'bands': [1, 3], 'nfeatures': 1}, {'type': 'glcm', 'bands': [1], 'features': ['homogeneity', 'energy']}]
c.ClusterManager.modelid = "kmeans"
c.ClusterManager.nclusters = 7
c.ClusterManager.random_state = 0
c.ClassificationManager.mid = "cnn2d"
c.ClassificationManager.nfeatures = 32
