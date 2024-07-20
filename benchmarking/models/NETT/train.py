import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ...glob import architectures, datasets, operators, solver, utilities


def image_artifacts(K, x, noise_level, p):
    # Save shape for later
    N, c, nx, ny = x.shape

    z_batch = np.zeros_like(x)
    r_batch = np.zeros_like(x)
    for i in range(N):
        # Get corresponding x_true
        x_true = x[i].numpy()

        # Compute x_dagger?
        coin_toss = np.random.binomial(n=1, p=p)
        if coin_toss == 0:
            z = x_true
        elif coin_toss == 1:
            # Flatten + Corrupt
            x_true_flat = x_true.flatten()
            y = K(x_true_flat)
            y_delta = y + utilities.gaussian_noise(y, noise_level=noise_level)

            # Compute z = x_dagger
            CGLS = solver.CGLS(K)
            z = CGLS(
                y_delta, x0=np.zeros_like(x_true_flat), x_true=x_true_flat, kmax=20
            ).reshape((c, nx, ny))

        z_batch[i] = z
        r_batch[i] = np.abs(x_true - z)
    return torch.tensor(z_batch), torch.tensor(r_batch)


###############################
# PARAMETERS
###############################
DATASET = "Mayo"  # in {"COULE", "Mayo"}

BATCH_SIZE = 4
N_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

p = 0.5  # Corruption probability

# Define paths
GT_PATH = f"../data/{DATASET}/train/"

# Setup inverse problem
nx, ny = 256, 256

ANGULAR_RANGE = 180
N_ANGLES = 60
DET_SIZE = int(max(nx, ny) * 2)
GEOMETRY = "fanflat"

NOISE_LEVEL = 0.01

# Get dataloader
train_data = datasets.ImageDataset(GT_PATH, transforms=transforms.Resize((nx, ny)))
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

######################## Define forward problem parameters
# Define the operator K
angles = np.linspace(
    np.deg2rad(0),
    np.deg2rad(ANGULAR_RANGE),
    N_ANGLES,
    endpoint=False,
)

K = operators.CTProjector((nx, ny), angles, det_size=DET_SIZE, geometry=GEOMETRY)

######################## Define train parameters
# Get NN model
model_suffix = "UNet"
model = architectures.UNet(img_ch=1, output_ch=1).to(DEVICE)

# Loss function
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Cycle over the epochs
print(f"Training ResUNet model for {N_EPOCHS} epochs and batch size of {BATCH_SIZE}.")
loss_total = np.zeros((N_EPOCHS,))
ssim_total = np.zeros((N_EPOCHS,))
for epoch in range(N_EPOCHS):

    # Cycle over the batches
    epoch_loss = 0.0
    ssim_loss = 0.0

    # Initialize tqdm
    loop = tqdm(train_loader)
    loop.set_description(f"Epoch: {epoch+1}/{N_EPOCHS} ->")
    for t, x_true in enumerate(loop):
        # Get artifact image
        x_in, artifacts = image_artifacts(K, x_true, noise_level=NOISE_LEVEL, p=p)

        # Send x and y to gpu
        x_in = x_in.to(DEVICE)
        artifacts = artifacts.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred_artifacts = model(x_in)
        loss = loss_fn(pred_artifacts, artifacts)
        loss.backward()
        optimizer.step()

        # update loss
        epoch_loss = epoch_loss + loss.item()
        loop.set_postfix(loss=epoch_loss / (t + 1))

        # Every 10 epochs, save the model weights
        if (epoch % 10) == 0:
            weights_path = f"./model_weights/NETT/{DATASET}{nx}_{ANGULAR_RANGE}_{N_ANGLES}_{model_suffix}.pth"
            torch.save(model.state_dict(), weights_path)

    # Update the history
    loss_total[epoch] = epoch_loss / (t + 1)
    ssim_total[epoch] = ssim_loss / (t + 1)

# Save the weights
weights_path = (
    f"./model_weights/NETT/{DATASET}{nx}_{ANGULAR_RANGE}_{N_ANGLES}_{model_suffix}.pth"
)
torch.save(model.state_dict(), weights_path)
