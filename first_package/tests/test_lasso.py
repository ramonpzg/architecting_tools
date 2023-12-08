import pytest
import numpy as np
from lassoreg.regression import LassoRegression
from hypothesis import given, strategies as st

# Example strategy for generating random data
@st.composite
def generate_random_data(draw):
    n_samples = draw(st.integers(min_value=1, max_value=100))
    n_features = draw(st.integers(min_value=1, max_value=10))
    X = draw(st.lists(st.lists(st.floats(), min_size=n_features, max_size=n_features), min_size=n_samples, max_size=n_samples))
    y = draw(st.lists(st.floats(), min_size=n_samples, max_size=n_samples))
    return np.array(X), np.array(y)

def test_lasso_regression_fit():
    # Test if the LassoRegression model can fit to synthetic data
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([5, 6])
    model = LassoRegression(alpha=0.01, max_iter=1000, tol=1e-4)
    model.fit(X_train, y_train)
    assert model.weights is not None

@given(generate_random_data())
def test_lasso_regression_convergence(random_data):
    # Test if the LassoRegression model converges with random data
    X, y = random_data
    model = LassoRegression(alpha=0.01, max_iter=1000, tol=1e-4)
    model.fit(X, y)
    assert model.weights is not None

def test_lasso_regression_predict():
    # Test if the LassoRegression model can make predictions
    X_test = np.array([[1, 2]])
    model = LassoRegression(alpha=0.01, max_iter=1000, tol=1e-4)
    model.weights = np.array([0.5, 0.2, 0.3])  # Sample weights for testing
    predictions = model.predict(X_test)
    assert predictions is not None