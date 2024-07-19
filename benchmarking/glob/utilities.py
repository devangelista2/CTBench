import numpy as np


def normalize(x):
    """Given an array x, returns its normalized version (i.e. the linear projection into [0, 1])."""
    return (x - x.min()) / (x.max() - x.min())


# Noise is added by noise level
def gaussian_noise(y, noise_level):
    e = np.random.randn(*y.shape)
    return e / np.linalg.norm(e.flatten()) * np.linalg.norm(y.flatten()) * noise_level
