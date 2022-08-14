import pytest
from itertools import product
import numpy as np
from sklearn.linear_model import LogisticRegression

from py_numba_blitz.solver import py_blitz
from skglm.utils import make_correlated_data


@pytest.mark.parametrize("n_samples, n_features, X_density",
                         product([10], [20], [1., 0.5]))
def test_alpha_max(n_samples, n_features, X_density):
    X, y, _ = make_correlated_data(n_samples, n_features,
                                   random_state=0, X_density=X_density)
    y = np.sign(y)

    alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / 2

    w = py_blitz(alpha_max, X, y)
    np.testing.assert_array_equal(w, 0)


@pytest.mark.parametrize("rho, X_density", product([1e-1, 1e-2, 1e-3], [1., 0.4]))
def test_vs_sklearn(rho, X_density):
    n_samples, n_features = 100, 200
    X, y, _ = make_correlated_data(n_samples, n_features,
                                   random_state=0, X_density=X_density)
    y = np.sign(y)

    alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / 2
    alpha = rho * alpha_max

    # sklearn model
    sk_logreg = LogisticRegression(solver='liblinear', penalty='l1', C=1/alpha,
                                   fit_intercept=False, tol=1e-9)
    sk_logreg.fit(X, y)

    # py blitz
    w = py_blitz(alpha, X, y, tol=1e-9)

    np.testing.assert_allclose(w, sk_logreg.coef_.flatten(), atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    pass
