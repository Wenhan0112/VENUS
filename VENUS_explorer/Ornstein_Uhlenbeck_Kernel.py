import numpy as np
import sklearn.gaussian_process as sk_gp

class Ornstein_Uhlenbeck_Kernel(sk_gp.kernel.Kernel):
    def __init__(self, l):
        self.l = l

    def _f(self, x, y):
        return np.exp(-np.linalg.norm(x - y) / self.l)

    def __call__(x, y=None, eval_gradient=False):
        if y is None:
            y = x

        result = np.array([[self._f(x0, y0) for y0 in y] for x0 in x])
        if eval_gradient:
            print("Warning: eval_gradient is not implemented for "
                "Ornstein_Uhlenbeck_Kernel")
            return result, np.zero_like(result)
        return result

    def diag(self, x):
        return np.ones(x.shape[0])

    def is_stationary(self):
        return True
