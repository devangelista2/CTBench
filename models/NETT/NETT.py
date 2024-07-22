import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from miscellanous import architectures, datasets, metrics, solver, utilities


class NETT:
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
            f"./model_weights/NETT/{self.dataset}{self.nx}_{config['angular_range']}_"
            + f"{config['n_angles']}_{self.model_suffix}.pth"
        )
        self.model = architectures.UNet(img_ch=self.n_ch, output_ch=self.n_ch).to(
            self.device
        )

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

    def line_search_F(self, x, y_delta):
        # Define f and grad_f
        f = lambda x_next: np.sum(np.square(self.K(x_next) - y_delta))
        grad_f = self.K.T(self.K(x) - y_delta)

        c = 1e-4
        rho = 0.9
        alpha = 1
        while f(x - alpha * grad_f) > f(x) - c * alpha * np.sum(np.square(grad_f)):
            alpha = rho * alpha
        return alpha

    def line_search_R(self, x, grad_f):
        # Define f and grad_f
        def f(x_next):
            with torch.no_grad():
                fx = torch.sum(torch.square(self.model(x_next)))
            return fx

        c = 1e-4
        rho = 0.9
        alpha = 1
        while f(x - alpha * grad_f) > f(x) - c * alpha * torch.sum(
            torch.square(grad_f)
        ):
            alpha = rho * alpha
        return alpha

    def train(self, batch_size=1, n_epochs=100, noise_level=0.01, p=0.5) -> None:
        # Get training data
        train_data = datasets.ImageDataset(
            self.gt_path, transforms=transforms.Resize((self.nx, self.ny))
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # Define loss function and optimizer
        loss_fn = nn.L1Loss()
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
                x_in, artifacts = self.image_artifacts(x_true, noise_level, p)

                # Send x and y to gpu
                x_in = x_in.to(self.device)
                artifacts = artifacts.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                pred_artifacts = self.model(x_in)
                loss = loss_fn(pred_artifacts, artifacts)
                loss.backward()
                optimizer.step()

                # update loss
                epoch_loss = epoch_loss + loss.item()
                ssim_loss = ssim_loss + metrics.SSIM_batch(pred_artifacts, artifacts)
                loop.set_postfix(loss=epoch_loss / (t + 1), ssim=ssim_loss / (t + 1))

            # Every 10 epochs, save the model weights
            if (epoch % 10) == 0:
                torch.save(self.model.state_dict(), self.weights_path)

            # Update the history
            loss_total[epoch] = epoch_loss / (t + 1)
            ssim_total[epoch] = ssim_loss / (t + 1)

        # Save the weights
        torch.save(self.model.state_dict(), self.weights_path)

    def image_artifacts(self, x, noise_level, p):
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
                y = self.K(x_true_flat)
                y_delta = y + utilities.gaussian_noise(y, noise_level=noise_level)

                # Compute z = x_dagger
                CGLS = solver.CGLS(self.K)
                z = CGLS(
                    y_delta, x0=np.zeros_like(x_true_flat), x_true=x_true_flat, kmax=20
                ).reshape((c, nx, ny))

            z_batch[i] = z
            r_batch[i] = np.abs(x_true - z)
        return torch.tensor(z_batch), torch.tensor(r_batch)

    def load_weights(self) -> None:
        self.model.load_state_dict(torch.load(self.weights_path))
