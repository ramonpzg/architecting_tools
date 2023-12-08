
from mlserver.codecs import decode_args
from mlserver.utils import get_model_uri
from mlserver import MLModel
from lassoreg.regression import LassoRegression
from sklearn import datasets
import numpy as np
import pickle
import os

class LassoCLS(MLModel):
    async def load(self) -> bool:
        X, y = datasets.load_diabetes(return_X_y=True)
        X = X[:150]
        y = y[:150]
        self.model = LassoRegression(alpha=1.0, max_iter=1000, tol=1e-4)
        self.model.fit(X, y)
        return True

    @decode_args
    async def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)
