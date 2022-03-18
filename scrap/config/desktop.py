c.ModeDataManager.data_dir = "/Volumes/Shared/Data/tess"
c.ModeDataManager.cache_dir = "/Volumes/Shared/Cache"
c.ModeDataManager.model_dims = 24
c.ModeDataManager.reduce_nepochs = 10
c.ModeDataManager.reduce_sparsity = 0.0
c.ModeDataManager.subsample_index = 1
c.ModeDataManager.reduce_method = "Autoencoder"
c.ModeDataManager.class_file = "NONE"
c.DataManager.proc_type = "cpu"
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
c.TextureManager.textures = [{'type': 'gabor', 'bands': [1, 3], 'nfeatures': 1}, {'type': 'glcm', 'bands': [1], 'features': ['homogeneity', 'energy']}]

