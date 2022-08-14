from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from py_numba_blitz.solver import py_blitz



class Solver(BaseSolver):

    name = "py_blitz"

    def __init__(self):
        pass

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.tol = 1e-9

        # Cache Numba compilation
        self.run(5)

    def run(self, n_iter):
        self.coef = py_blitz(self.lmbd, self.X, self.y, tol=self.tol,
                             max_epochs=1000, max_iter=n_iter)

    def get_result(self):
        return self.coef