from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, minmax_scale
from typing import List, Union, Dict, Callable, Tuple, Optional
from sklearn.svm import LinearSVC
import xarray as xa
import time, traceback
from typing import List, Tuple, Optional, Dict
import numpy as np
from .base import LearningModel

class SVCLearningModel(LearningModel):

    def __init__(self, **kwargs ):
        norm = kwargs.pop( 'norm', True )
        tol = kwargs.pop( 'tol', 1e-5 )
        LearningModel.__init__(self, "svc",  **kwargs )
        self._score: Optional[np.ndarray] = None
        if norm: self.svc = make_pipeline( StandardScaler(), LinearSVC( tol=tol, dual=False, fit_intercept=False, **kwargs ) )
        else:    self.svc = LinearSVC(tol=tol, dual=False, fit_intercept=False, **kwargs)

    def fit( self, X: np.ndarray, y: np.ndarray, **kwargs ):       # X[n_samples, n_features], y[n_samples]
        t0 = time.time()
        print(f"Running SVC fit, X shape: {X.shape}), y shape: {y.shape})")
        self.svc.fit( X, y )
        self._score = self.decision_function(X)
        print(f"Completed SVC fit, in {time.time()-t0} secs")

#        self._support_vector_indices = np.where( (2 * y - 1) * self._score <= 1 )[0]    # For binary classifier
#        self._support_vectors = X[ self.support_vector_indices ]

    def predict( self, X: np.ndarray, **kwargs ) -> np.ndarray:
        return self.svc.predict( X ).astype( int )

    @property
    def decision_function(self) -> Callable:
        return self.svc.decision_function



