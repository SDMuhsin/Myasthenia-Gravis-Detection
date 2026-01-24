"""
ROCKET Wrapper for SOTA Comparison

ROCKET (RandOm Convolutional KErnel Transform) is a fast and accurate method
for time series classification using random convolutional kernels.

Reference:
    Dempster et al. "ROCKET: Exceptionally fast and accurate time series
    classification using random convolutional kernels" (DMKD 2020)
"""

import numpy as np
import time
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import BaselineModel

try:
    from sktime.transformations.panel.rocket import Rocket
    ROCKET_AVAILABLE = True
except ImportError:
    ROCKET_AVAILABLE = False


class ROCKETWrapper(BaselineModel):
    """
    ROCKET: Random Convolutional Kernel Transform.

    This is a scikit-learn compatible model that uses:
    1. ROCKET transform to extract features using random convolutional kernels
    2. Logistic regression classifier for final classification

    For long sequences, subsampling is applied to reduce computational cost.

    Reference:
        Dempster et al. "ROCKET: Exceptionally fast and accurate time series
        classification using random convolutional kernels" (2020)
    """

    def __init__(
        self,
        input_dim: int = 14,
        num_classes: int = 2,
        seq_len: int = 290,
        num_kernels: int = 2000,
        max_seq_len: int = 500,  # Maximum sequence length before subsampling
        random_state: int = 42,
        **kwargs,
    ):
        if not ROCKET_AVAILABLE:
            raise ImportError(
                "ROCKET requires sktime. Install with: pip install sktime"
            )

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.num_kernels = num_kernels
        self.max_seq_len = max_seq_len
        self.random_state = random_state

        # Calculate subsampling stride
        self.stride = max(1, seq_len // max_seq_len)
        self.effective_seq_len = (seq_len + self.stride - 1) // self.stride

        # ROCKET transform
        self.rocket = Rocket(num_kernels=num_kernels, random_state=random_state)

        # Feature scaler
        self.scaler = StandardScaler()

        # Logistic regression with regularization
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver='lbfgs',
            n_jobs=-1,  # Use all cores
            random_state=random_state,
        )

        self._is_fitted = False
        self._n_features = None

    def _subsample(self, X: np.ndarray) -> np.ndarray:
        """
        Subsample time series to reduce length.

        Args:
            X: Input array (n_samples, seq_len, n_features)

        Returns:
            Subsampled array (n_samples, reduced_seq_len, n_features)
        """
        if self.stride > 1:
            return X[:, ::self.stride, :]
        return X

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Fit the ROCKET model.

        Args:
            X_train: Training data (n_samples, seq_len, n_features)
            y_train: Training labels (n_samples,)
        """
        start_time = time.time()

        # Subsample for efficiency
        X_train_sub = self._subsample(X_train)

        # Convert to sktime format: (n_samples, n_features, seq_len)
        # sktime expects multivariate time series as (n_instances, n_dimensions, series_length)
        X_train_sktime = np.transpose(X_train_sub, (0, 2, 1))

        # Fit ROCKET transform
        X_transform = self.rocket.fit_transform(X_train_sktime)

        # Scale features
        X_transform_scaled = self.scaler.fit_transform(X_transform)

        # Fit classifier
        self.classifier.fit(X_transform_scaled, y_train)

        self._is_fitted = True
        self._n_features = X_transform.shape[1]

        training_time = time.time() - start_time

        return {
            'training_time': training_time,
            'best_epoch': 0,  # N/A for ROCKET
            'best_val_loss': 0.0,
            'n_rocket_features': self._n_features,
            'subsample_stride': self.stride,
            'effective_seq_len': self.effective_seq_len,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Subsample
        X_sub = self._subsample(X)

        # Convert to sktime format
        X_sktime = np.transpose(X_sub, (0, 2, 1))

        # Transform
        X_transform = self.rocket.transform(X_sktime)
        X_transform_scaled = self.scaler.transform(X_transform)

        return self.classifier.predict(X_transform_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Subsample
        X_sub = self._subsample(X)

        # Convert to sktime format
        X_sktime = np.transpose(X_sub, (0, 2, 1))

        # Transform
        X_transform = self.rocket.transform(X_sktime)
        X_transform_scaled = self.scaler.transform(X_transform)

        # LogisticRegression has native predict_proba
        return self.classifier.predict_proba(X_transform_scaled)

    def count_parameters(self) -> int:
        """
        Return number of 'parameters'.

        For ROCKET, this is the number of kernels * 2 (for PPV and max features)
        plus the classifier coefficients.
        """
        n_params = self.num_kernels * 2  # PPV and max features per kernel
        if self._is_fitted and hasattr(self.classifier, 'coef_'):
            n_params += self.classifier.coef_.size
        return n_params

    def get_model_name(self) -> str:
        return "ROCKET"
