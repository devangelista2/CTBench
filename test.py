import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

import config
from miscellanous import datasets, metrics, utilities
from models import ResidualUnfold
from torch_utils import operators as torchops
from torch_utils import utilities as tutils

##################### PARAMETER INITIALIZATION
# Define dataset
dataset = "Mayo"  # Mayo, COULE
idx = 10  # 10 or 6

# Define additional problem parameters
noise_level = 0.01  # Mayo 0.001, COULE 0.01

# Load configuration
cfg = config.initialize_default_config(dataset)
# cfg = config.parse_config(cfg)

img_ch, nx, ny = cfg["image_shape"]
gt_path = f"../data/{dataset}/train/"
device = cfg["device"]

# Load test data and x_true
test_data = datasets.ImageDataset(gt_path, transforms=transforms.Resize((nx, ny)))
x_true = test_data[idx].unsqueeze(0).to(device)

# Initialize test problem
K = tutils.initialize_CT_projector(cfg)

y = K(x_true)
y_delta = y + tutils.gaussian_noise(y, noise_level)

###### LOAD MODEL
model = ResidualUnfold.ResidualUnfold(cfg)
model.train(batch_size=1, n_epochs=10, noise_level=0.01)
