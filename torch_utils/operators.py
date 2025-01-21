import math

import torch

from torch_utils import utilities


class Radon:
    def __init__(self, input_shape, angles, det_size=None, geometry="parallel") -> None:
        # Input setup
        self.input_shape = input_shape
        self.N, self.c, self.h, self.w = input_shape

        # Geometry
        self.geometry = geometry

        # Projector setup
        if det_size is None:
            self.det_size = int(self.h * math.sqrt(2))
        else:
            self.det_size = det_size
        self.angles = angles
        self.n_angles = len(angles)

        # Define projector
        self.proj = self.get_astra_projection_operator()
        self.shape = self.proj.shape

    def __call__(self, x):
        # Assert the number of channels of x is 1
        assert x.shape[1] == 1

        # Save x device
        x_device = x.device

        self.K = utilities.CustomNumpyOperator()
        y = self.K.apply(self.proj, x[0, 0].cpu().flatten())
        return y.to(x_device)

    def T(self, y):
        # Save y device
        y_device = y.device

        self.K = utilities.CustomNumpyOperator()
        y = self.K.apply(self.proj.T, y.cpu().flatten())
        return y.reshape(self.input_shape).to(y_device)

    def get_astra_projection_operator(self):
        import astra

        # create geometries and projector
        if self.geometry == "parallel":
            proj_geom = astra.create_proj_geom(
                "parallel", 1.0, self.det_size, self.angles
            )
            vol_geom = astra.create_vol_geom(self.h, self.w)
            proj_id = astra.create_projector("linear", proj_geom, vol_geom)

        elif self.geometry == "fanflat":
            proj_geom = astra.create_proj_geom(
                "fanflat", 1.0, self.det_size, self.angles, 1800, 500
            )
            vol_geom = astra.create_vol_geom(self.h, self.w)
            proj_id = astra.create_projector("cuda", proj_geom, vol_geom)

        else:
            raise NotImplementedError("Geometry (still) undefined.")

        return astra.OpTomo(proj_id)

    def FBP(self, y):
        x = self.proj.reconstruct("FBP_CUDA", y.numpy().flatten())
        return torch.tensor(x.reshape((1, 1, self.h, self.w)), requires_grad=False)
