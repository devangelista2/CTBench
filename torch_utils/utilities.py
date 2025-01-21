import numpy as np
import torch

from torch_utils import operators


########################
# PYTORCH UTILITIES
########################
class CustomNumpyOperator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, K, x):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.

        K -> Operator that can be applied to Numpy version of x. It requires a __call__ method and a .T.
        x -> Pytorch array to which K has to be applied.
        """
        ctx.save_for_backward(x)
        ctx.K = K

        x_npy = x.detach().numpy()
        y_npy = ctx.K(x_npy)
        y = torch.from_numpy(y_npy)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (x,) = ctx.saved_tensors

        grad_output_npy = grad_output.numpy()
        KT_grad_output_npy = ctx.K.T(grad_output_npy)
        KT_grad_output = torch.from_numpy(KT_grad_output_npy)
        return None, KT_grad_output


def initialize_CT_projector(config):
    # Extract informations
    _, nx, ny = config["image_shape"]
    angular_range = config["angular_range"]
    n_angles = config["n_angles"]
    det_size = config["det_size"]
    geometry = config["geometry"]
    angles = np.linspace(
        np.deg2rad(0), np.deg2rad(angular_range), n_angles, endpoint=False
    )

    # Define projector
    K = operators.Radon(
        input_shape=(1, 1, nx, ny), angles=angles, det_size=det_size, geometry=geometry
    )
    return K


# Noise is added by noise level
def gaussian_noise(y, noise_level):
    e = torch.randn(*y.shape, device=y.device)
    return e / torch.norm(e.flatten()) * torch.norm(y.flatten()) * noise_level
