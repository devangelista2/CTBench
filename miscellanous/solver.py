import numpy as np


class CGLS:
    def __init__(self, K):
        self.K = K

    def __call__(self, b, x0, x_true=None, kmax=100, tolf=1e-6, tolx=1e-6, info=False):
        d = b
        r0 = self.K.T(b)
        p = r0
        t = self.K(p)

        x = x0
        r = r0
        k = 0

        if x_true is not None:
            err_vec = np.zeros((kmax, 1))
            err_vec[k] = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

            condition = np.linalg.norm(r) > tolf and err_vec[k] > tolx and k < kmax - 1
        else:
            condition = np.linalg.norm(r) > tolf and k < kmax - 1
        while condition:
            x0 = x

            alpha = np.linalg.norm(r0, 2) ** 2 / np.linalg.norm(t, 2) ** 2
            x = x0 + alpha * p
            d = d - alpha * t
            r = self.K.T(d)
            beta = np.linalg.norm(r, 2) ** 2 / np.linalg.norm(r0, 2) ** 2
            p = r + beta * p
            t = self.K(p)
            k = k + 1

            r0 = r

            if x_true is not None:
                err_vec[k] = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
                condition = (
                    np.linalg.norm(r) > tolf and err_vec[k] > tolx and k < kmax - 1
                )

            else:
                condition = np.linalg.norm(r) > tolf and k < kmax - 1

        if x_true is not None:
            err_vec = err_vec[: k + 1]

            if info:
                return x, err_vec
        return x
