import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from miscellanous import architectures, datasets, operators, solver, utilities


class FISTAWavelet:
    def __init__(self, config) -> None:
        self.config = config

        self.n_ch, self.nx, self.ny = config["image_shape"]

        # Define additional parameters
        self.K = utilities.initialize_CT_projector(config)

    def __call__(
        self,
        y_delta,
        lmbda,
        x_0=None,
        x_true=None,
        step_size_F=1e-4,
        step_size_R=1e-2,
        maxit=100,
    ):
        if x_0 is None:
            x_0 = np.zeros((self.n_ch * self.nx * self.ny,))

        for k in range(maxit):
            # Fidelity update
            if step_size_F is None:
                step_size_F = self.line_search_F(x_0, y_delta)
                x_tilde = x_0 - step_size_F * self.K.T(self.K(x_0) - y_delta)
                step_size_F = None
            else:
                x_tilde = x_0 - step_size_F * self.K.T(self.K(x_0) - y_delta)

            ###### Regularizer update
            x_tilde = torch.tensor(
                x_tilde.reshape((1, self.n_ch, self.nx, self.ny)),
                requires_grad=True,
                device=self.device,
                dtype=torch.float32,
            )
            pred_artifacts = self.model(x_tilde)

            # Compute R(x)
            reg = torch.sum(torch.square(pred_artifacts))

            # Compute grad(R(x))
            reg.backward()

            # Update x
            if step_size_R is None:
                step_size_R = self.line_search_R(x_tilde, x_tilde.grad)
                x = x_tilde - step_size_R * lmbda * x_tilde.grad
                step_size_R = None
            else:
                x = x_tilde - step_size_R * lmbda * x_tilde.grad

            # Verbose informations
            gradF = np.linalg.norm(self.K.T(self.K(x.detach().cpu()) - y_delta))
            gradR = np.linalg.norm(x_tilde.grad.cpu().numpy().flatten())
            grad = np.linalg.norm(
                self.K.T(self.K(x.detach().cpu().numpy().flatten()) - y_delta)
                + lmbda * x_tilde.grad.cpu().numpy().flatten()
            )

            if x_true is not None:
                RE = np.linalg.norm(
                    x.detach().cpu().flatten() - x_true.flatten()
                ) / np.linalg.norm(x_true.flatten())
            print(
                f"It. {k+1}/{maxit}: gradF = {gradF:.3f}, gradR = {gradR:.3f}, grad = {grad:.3f}, RE = {RE:.3f}."
            )

            # Restart
            x_tilde.grad = None
            x_0 = x.detach().cpu().numpy().flatten()

        return x.detach().cpu()

    def fista_wavelet_2d_operator(
        self, y_delta, lmbda, wavelet="db1", maxit=100, tol=1e-6
    ):
        # Initialize variables
        x_k = np.zeros_like(y_delta)
        x_k_1 = x_k.copy()
        t_k = 1
        t_k_1 = 1

        for i in range(maxit):
            # Compute the gradient of the data fidelity term
            grad = self.K.T(self.K(x_k) - y_delta)

            # Gradient descent step
            x_k = x_k - grad

            # Perform wavelet transform
            coeffs = pywt.wavedec2(x_k, wavelet)
            coeffs_flat, slices = pywt.coeffs_to_array(coeffs)

            # Soft-thresholding
            coeffs_flat = self.soft_thresholding(coeffs_flat, lmbda)

            # Reshape to original wavelet coefficient shape
            coeffs_reconstructed = pywt.array_to_coeffs(
                coeffs_flat, slices, output_format="wavedec2"
            )

            # Inverse wavelet transform
            x_k = pywt.waverec2(coeffs_reconstructed, wavelet)

            # Update step
            t_k = (1 + np.sqrt(1 + 4 * t_k_1**2)) / 2
            x_k = x_k + ((t_k_1 - 1) / t_k) * (x_k - x_k_1)

            # Check convergence
            if np.linalg.norm(x_k - x_k_1) / np.linalg.norm(x_k_1) < tol:
                break

            x_k_1 = x_k.copy()
            t_k_1 = t_k

        return x_k

    def soft_thresholding(self, x, lmbda):
        return np.sign(x) * np.maximum(np.abs(x) - lmbda, 0)
