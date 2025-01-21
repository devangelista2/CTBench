# This is an implementation from the paper "Deep residual unfolding: A novel sparse computed tomography reconstruction method leveraging
# iterative learning and neural networks" which however contains multiple uncertainties.
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from miscellanous import architectures, datasets, metrics, utilities
from torch_utils import utilities as torch_utilities


class ResidualUnfold:
    def __init__(self, config) -> None:
        self.config = config

        # Load informations from config
        self.device = config["device"]

        self.dataset = config["dataset"]
        self.n_ch, self.nx, self.ny = config["image_shape"]

        # Define additional parameters
        self.model_suffix = "ResidualUnfold"
        self.gt_path = f"../data/{self.dataset}/train/"

        self.K = torch_utilities.initialize_CT_projector(config)

        self.weights_path = (
            f"./model_weights/ResidualUnfold/{self.dataset}{self.nx}_{config['angular_range']}_"
            + f"{config['n_angles']}_{self.model_suffix}.pth"
        )
        self.model = architectures.ResidualUnfoldBlock(img_ch=1, n_ch=64).to(
            self.device
        )

    def __call__(
        self,
        y_delta,
        beta=1e-3,
        maxit=20,
    ) -> torch.Tensor:
        # Preprocess y_delta -> FBP(y_delta)
        x = self.K.FBP(y_delta.cpu()).to(self.device)

        # Cycle over iterations
        for k in range(maxit):
            ### First path: model-based
            x_MB = x - beta * self.K.T(self.K(x) - y_delta)

            ### Second path: data-based
            x_DB = self.model(x)

            # Update x
            x = x_MB + x_DB

        return x

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
                # Get corrputed y_delta
                y = self.K(x_true)
                y_delta = y + torch_utilities.gaussian_noise(y, noise_level=noise_level)

                # Send x and y to gpu
                x_true = x_true.to(self.device)
                y_delta = y_delta.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                x_pred = self.__call__(y_delta, beta=1e-4, maxit=4)
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

    def load_weights(self) -> None:
        self.model.load_state_dict(torch.load(self.weights_path))
