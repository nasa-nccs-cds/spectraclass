from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, minmax_scale
from typing import List, Union, Dict, Callable, Tuple, Optional
from sklearn.svm import LinearSVC, SVC
import xarray as xa
import time, traceback
from typing import List, Tuple, Optional, Dict
import numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from .base import LearningModel
from spectraclass.learn.base import ModelType

class SVCLearningModel(LearningModel):

    def __init__(self, **kwargs ):
        self.norm = kwargs.pop( 'norm', True )
        self.tol = kwargs.pop( 'tol', 1e-5 )
        self.kernel = kwargs.pop('kernel', "linear" ) # curve rbf linear
        LearningModel.__init__(self, "svc", ModelType.MODEL,  **kwargs )
        self.parms  = kwargs
        self._score: Optional[np.ndarray] = None
        self.create_model()

    def create_model(self):
        model = SVC( tol=self.tol, kernel=self.kernel, probability=True, **self.parms )
        if self.norm: self.svc = make_pipeline( StandardScaler(), model  )
        else:  self.svc = model

    def clear(self):
        self.create_model()

    def fit( self, X: np.ndarray, y: np.ndarray, **kwargs ):       # X[n_samples, n_features], y[n_samples]
        t0 = time.time()
        lgm().log(f"Running SVC fit, X shape: {X.shape}), y shape: {y.shape})")
        self.svc.fit( X, y )
        self._score = self.decision_function(X)
        lgm().log(f"Completed SVC fit, in {time.time()-t0} secs")

#        self._support_vector_indices = np.where( (2 * y - 1) * self._score <= 1 )[0]    # For binary classifier
#        self._support_vectors = X[ self.support_vector_indices ]

    def predict( self, X: np.ndarray, **kwargs ) -> np.ndarray:
        t0 = time.time()
        lgm().log(f"Running SVC predict, X shape: {X.shape})")
        result = self.svc.predict( X ).astype( int )
        lgm().log(f"Completed SVC predict, in {time.time() - t0} secs, result shape = {result.shape}")
        return result

    def probability( self, X: np.ndarray ) -> np.ndarray:
        return self.svc.predict_proba( X )

    @property
    def decision_function(self) -> Callable:
        return self.svc.decision_function



