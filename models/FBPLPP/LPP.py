import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from miscellanous import architectures, datasets, metrics, utilities


class FBP_LPP:
    def __init__(self, config) -> None:
        self.config = config

        # Load informations from config
        self.device = config["device"]

        self.dataset = config["dataset"]
        self.n_ch, self.nx, self.ny = config["image_shape"]

        # Define additional parameters
        self.model_suffix = "UNet"
        self.gt_path = f"../data/{self.dataset}/train/"

        self.K = utilities.initialize_CT_projector(config)

        self.weights_path = (
            f"./model_weights/FBP-LPP/{self.dataset}{self.nx}_{config['angular_range']}_"
            + f"{config['n_angles']}_{self.model_suffix}.pth"
        )
        self.model = architectures.ResUNet(img_ch=self.n_ch, output_ch=self.n_ch).to(
            self.device
        )

    def __call__(
        self,
        y_delta,
    ):
        # Preprocess y_delta -> FBP(y_delta)
        x_FBP = self.K.FBP(y_delta)

        # Convert x_FBP to torch tensor and send it to device
        x_FBP = torch.tensor(
            x_FBP.reshape((1, 1, self.nx, self.ny)),
            device=self.device,
            requires_grad=False,
        )

        # Compute x_LPP
        with torch.no_grad():
            x_LPP = self.model(x_FBP)

        return x_LPP.cpu()

    def train(self, batch_size=1, n_epochs=100, noise_level=0.01) -> None:
        # Get training data
        train_data = datasets.ImageDataset(
            self.gt_path, transforms=transforms.Resize((self.nx, self.ny))
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # Define loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Cycle over the epochs
        print(
            f"Training {self.model_suffix} model for {n_epochs} epochs and batch size of {batch_size}."
        )
        loss_total = np.zeros((n_epochs,))
        ssim_total = np.zeros((n_epochs,))
        for epoch in range(n_epochs):

            # Cycle over the batches
            epoch_loss = 0.0
            ssim_loss = 0.0

            # Initialize tqdm
            loop = tqdm(train_loader)
            loop.set_description(f"Epoch: {epoch+1}/{n_epochs} ->")
            for t, x_true in enumerate(loop):
                # Get artifact image
                x_FBP = self.corrupt_batch(x_true, noise_level)

                # Send x and y to gpu
                x_FBP = x_FBP.to(self.device)
                x_true = x_true.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                x_pred = self.model(x_FBP)
                loss = loss_fn(x_pred, x_true)
                loss.backward()
                optimizer.step()

                # update loss
                epoch_loss = epoch_loss + loss.item()
                ssim_loss = ssim_loss + metrics.SSIM_batch(x_pred, x_true)
                loop.set_postfix(loss=epoch_loss / (t + 1), ssim=ssim_loss / (t + 1))

            # Every 10 epochs, save the model weights
            if (epoch % 10) == 0:
                torch.save(self.model.state_dict(), self.weights_path)

            # Update the history
            loss_total[epoch] = epoch_loss / (t + 1)
            ssim_total[epoch] = ssim_loss / (t + 1)

        # Save the weights
        torch.save(self.model.state_dict(), self.weights_path)

    def corrupt_batch(self, x, noise_level):
        x_FBP = np.zeros_like(x)
        for i in range(x.shape[0]):
            # Get corresponding x_true
            x_true = x[i].numpy()

            # Flatten + Corrupt
            x_true_flat = x_true.flatten()
            y = self.K(x_true_flat)
            y_delta = y + utilities.gaussian_noise(y, noise_level=noise_level)

            # Compute FBP(y_delta)
            x_FBP[i] = self.K.FBP(y_delta).reshape((1, self.n_ch, self.nx, self.ny))

        return torch.tensor(x_FBP)

    def load_weights(self) -> None:
        self.model.load_state_dict(torch.load(self.weights_path))
