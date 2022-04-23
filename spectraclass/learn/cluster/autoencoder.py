from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

class AutoEncoderCluster(TransformerMixin,ClusterMixin,BaseEstimator):

    def __init__( self, n_clusters=8, *, init="auto", max_iter=100 ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter

