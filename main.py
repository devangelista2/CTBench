import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

import config
from miscellanous import datasets, solver, utilities
from models.FBPLPP import LPP
from models.NETT import NETT
from models.FISTA import FISTA

##################### TEST SETUP
NETT_test = False
FBPLPP_test = False
FISTA_test = True

##################### PARAMETER INITIALIZATION
# Define dataset
dataset = "Mayo"
idx = 10

# Define additional problem parameters
noise_level = 0.01

# Load configuration
cfg = config.initialize_default_config(dataset)
cfg = config.parse_config(cfg)

img_ch, nx, ny = cfg["image_shape"]
gt_path = f"../data/{dataset}/test/"
device = cfg["device"]

# Load test data and x_true
test_data = datasets.ImageDataset(gt_path, transforms=transforms.Resize((nx, ny)))
x_true = test_data[idx]

# Initialize test problem
K = utilities.initialize_CT_projector(cfg)

y = K(x_true)
y_delta = y + utilities.gaussian_noise(y, noise_level)

################## NETT Testing
if NETT_test:
    NETT_model = NETT.NETT(cfg)
    NETT_model.load_weights()

    x_NETT = NETT_model(
        y_delta, lmbda=5, x_true=x_true, step_size_F=1e-4, step_size_R=1e-3, maxit=300
    )

################## FBP-LPP Testing
if FBPLPP_test:
    FBPLPP_model = LPP.FBP_LPP(cfg)
    FBPLPP_model.load_weights()

    x_FBPLPP = FBPLPP_model(y_delta)

################## FISTA-W Testing
if FISTA_test:
    FISTA_model = FISTA.FISTAWavelet(cfg)

    x_FISTA = FISTA_model(y_delta, lmbda=0.10, x_true=x_true, maxit=100)