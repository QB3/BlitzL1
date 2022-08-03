import blitzl1
import numpy as np
from skglm.utils import make_correlated_data

n_samples, n_features = 500, 2000
rho = 0.05

X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
y = np.sign(y)


alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / 2
alpha = rho * alpha_max
max_iter = 100

blitzl1.set_use_intercept(False)
blitzl1.set_tolerance(0)
problem = blitzl1.LogRegProblem(X, y)

  
coef_ = problem.solve(alpha, max_iter=max_iter).x
coef_.flatten()

# print(coef_[coef_ != 0])