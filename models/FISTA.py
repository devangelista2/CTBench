import numpy as np
import pywt

from miscellanous import utilities


class FISTAWavelet:
    def __init__(self, config) -> None:
        self.config = config

        self.n_ch, self.nx, self.ny = config["image_shape"]

        # Define additional parameters
        self.K = utilities.initialize_CT_projector(config)

        # Compute Lipschitz constant of K.T K
        self.L = self.power_method(maxit=10)

    def __call__(
        self, 
        y_delta, 
        lmbda,
        x_0=None,
        x_true=None,
        wavelet_type="db1", 
        maxit=100, 
        tolx=1e-6,
    ) -> np.ndarray:
        # Initialization
        if x_0 is None:
            x_0 = np.zeros((self.nx * self.ny,))
        x = x_0.copy()

        # Step number
        alpha = 1 / self.L
        k = 0

        # Initialize t_k
        t_k = 1
        t_k_1 = 1
        
        stopping_condition = True
        while stopping_condition:
            # Gradient descent step
            x = x - alpha * self.K.T(self.K(x) - y_delta)

            # Perform wavelet transform
            a, slices = pywt.coeffs_to_array(pywt.wavedec2(x.reshape((self.nx, self.ny)), wavelet_type))

            # Soft-thresholding
            a = self.soft_thresholding(a, lmbda / t_k)

            # Reshape to original wavelet coefficient shape
            coeffs_reconstructed = pywt.array_to_coeffs(
                a, slices, output_format="wavedec2"
            )

            # Inverse wavelet transform
            x = pywt.waverec2(coeffs_reconstructed, wavelet_type).flatten()

            # Update step
            t_k = (1 + np.sqrt(1 + 4 * t_k_1**2)) / 2
            x = x + ((t_k_1 - 1) / t_k) * (x - x_0)
            k = k + 1

            if x_true is not None:
                RE = np.linalg.norm(
                    x.flatten() - x_true.numpy().flatten()
                ) / np.linalg.norm(x_true.numpy().flatten())
                print(
                    f"It. {k}/{maxit}: RE = {RE:.3f}."
                )  
            
            # Check convergence
            stopping_condition = (k < maxit) and (np.linalg.norm(x - x_0) > tolx * np.linalg.norm(x_0))

            # Restart
            x_0 = x.copy()
            t_k_1 = t_k

        return x

    def soft_thresholding(self, x, lmbda):
        return np.sign(x) * np.maximum(np.abs(x) - lmbda, 0)
    
    def power_method(self, maxit=10):
        """
        Calculates the norm of operator K.T K,

        K : forward projection
        maxit : number of iterations to perform (default: 10)
        """
        x = np.random.rand(self.nx, self.ny)
        for _ in range(maxit):
            x = self.K.T(self.K(x))
            x_norm = np.linalg.norm(x.flatten(), 2)
            x = x / x_norm
        return x_norm
