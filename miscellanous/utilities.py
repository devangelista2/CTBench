import numpy as np

from miscellanous import operators


def normalize(x):
    """Given an array x, returns its normalized version (i.e. the linear projection into [0, 1])."""
    return (x - x.min()) / (x.max() - x.min())


# Noise is added by noise level
def gaussian_noise(y, noise_level):
    e = np.random.randn(*y.shape)
    return e / np.linalg.norm(e.flatten()) * np.linalg.norm(y.flatten()) * noise_level

def initialize_CT_projector(config):
    # Extract informations
    _, nx, ny = config["image_shape"]
    angular_range = config["angular_range"]
    n_angles = config["n_angles"]
    det_size = config["det_size"]
    geometry = config["geometry"]
    angles = np.linspace(np.deg2rad(0), np.deg2rad(angular_range), n_angles, endpoint=False)

    # Define projector
    K = operators.CTProjector((nx, ny), angles, det_size=det_size, geometry=geometry)
    return K