import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, default_collate
import matplotlib.pyplot as plt
torch.manual_seed(seed=42)


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*16*128, latent_dim * 2)  # μ and log(σ²)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16*16*128),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        # print('h:', h.shape)
        mu, logvar = h.chunk(2, dim=1)
        # print('mu:', mu.shape)
        # print('logvar:', logvar.shape)
        z = self.reparameterize(mu, logvar)
        # print(z.shape)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


# Loss = Reconstruction + KL Divergence
def vae_loss(x_recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print('recon_loss:', recon_loss, 'kl_loss:', kl_loss, 'tot:', recon_loss+kl_loss)
    return recon_loss + kl_loss


class WAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Same encoder/decoder as ConvAE
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*16*128, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16*16*128),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


def collate_to_gpu(batch, device):
    images, labels = torch.utils.data.default_collate(batch)
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def collate_to_cpu(batch):
    return default_collate(batch)  # Keeps data on CPU


# Training function
def train_vae(model, dataloader, epochs, device='cuda', latent_dim=128, optimizer=None):
    model.train()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        recon_losses = 0
        kl_losses = 0

        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)

            # Forward pass
            x_recon, mu, logvar = model(x)

            # Compute loss
            loss = vae_loss(x_recon, x, mu, logvar)
            recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track losses
            total_loss += loss.item()
            recon_losses += recon_loss.item()
            kl_losses += kl_loss.item()

            # Log every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} "
                      f"| Loss: {loss.item()/len(x):.4f} "
                      f"(Recon: {recon_loss.item()/len(x):.4f}, KL: {kl_loss.item()/len(x):.4f})")

        # Epoch statistics
        avg_loss = total_loss / len(dataloader.dataset)
        avg_recon = recon_losses / len(dataloader.dataset)
        avg_kl = kl_losses / len(dataloader.dataset)

        print(f"=== Epoch {epoch+1} Summary ===")
        print(f"Avg Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}\n")

        """
        # Save generated samples (optional)
        if epoch % 5 == 0:
            with torch.no_grad():
                sample = torch.randn(16, latent_dim).to(device)
                generated = model.decoder(sample).cpu()
        """

    checkpoint = {'epochs': epochs,  'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'vae_'+str(epochs).zfill(4)+'.pth')

    return model


def compute_sigma(z):
    pairwise_dist = torch.cdist(z, z)
    return torch.median(pairwise_dist).item()


def mmd_loss_multiscale(z, prior, sigmas=[0.1, 1.0, 10.0]):
    loss = 0.0
    for sigma in sigmas:
        loss += mmd_loss(z, prior, sigma=sigma)
    return loss / len(sigmas)


# Training function (WAE-MMD)
def train_wae(model, dataloader, device='cuda', epochs=2, latent_dim=128,  checkpoint=None):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs_done = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_done = checkpoint['epochs']
    model.train()

    for epoch in range(epochs_done, epochs):
        total_loss = 0
        recon_losses = 0
        mmd_losses = 0

        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)

            # Forward pass
            x_recon, z = model(x)
            prior_samples = torch.randn_like(z)  # Sample from N(0,1)

            # Dynamic sigma (median heuristic)
            sigma = compute_sigma(z) if epoch > 0 else 1.0

            # Compute losses
            recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
            # mmd = mmd_loss(z, prior_samples)
            mmd = mmd_loss_multiscale(z, prior_samples, sigmas=[0.1, 1.0, 10.0])
            # loss = mmd
            loss = recon_loss * torch.abs(mmd)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track losses
            total_loss += loss.item()
            recon_losses += recon_loss.item()
            mmd_losses += mmd.item()

            # Log progress
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} "
                      f"| Loss: {loss.item()/len(x):.4f} "
                      f"(Recon: {recon_loss.item()/len(x):.4f}, MMD: {mmd.item():.6f})")

        # Epoch summary
        avg_loss = total_loss / len(dataloader.dataset)
        avg_recon = recon_losses / len(dataloader.dataset)
        avg_mmd = mmd_losses / len(dataloader)

        print(f"=== Epoch {epoch+1} Summary ===")
        print(f"Avg Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | MMD: {avg_mmd:.4f}\n")

        """
        # Generate samples
        if epoch % 5 == 0:
            with torch.no_grad():
                z_sample = torch.randn(16, latent_dim).to(device)
                generated = model.decoder(z_sample).cpu()
                writer.add_images('Generated', (generated + 1) / 2, epoch)
        """

    checkpoint = {'epochs': epochs,  'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'wae_'+str(epochs).zfill(4)+'.pth')

    return model


# MMD Loss (RBF kernel)
def mmd_loss(z, prior_samples, sigma=1.0):
    zz = torch.matmul(z, z.t())
    zp = torch.matmul(z, prior_samples.t())
    pp = torch.matmul(prior_samples, prior_samples.t())

    K_zz = torch.exp(-0.5 * (zz.diag().unsqueeze(1) + zz.diag().unsqueeze(0) - 2 * zz) / sigma**2)
    K_zp = torch.exp(-0.5 * (zz.diag().unsqueeze(1) + pp.diag().unsqueeze(0) - 2 * zp) / sigma**2)
    K_pp = torch.exp(-0.5 * (pp.diag().unsqueeze(1) + pp.diag().unsqueeze(0) - 2 * pp) / sigma**2)

    return K_zz.mean() - 2 * K_zp.mean() + K_pp.mean()


def interpolate(model, x1, x2, n=10, tag='ae', device='cuda'):
    """
  Interpolate between two input images and display the sequence using OpenCV.
   Args:
        model: Trained AE/WAE model.
        x1, x2: Input images(PyTorch tensors, shape[1, 784]).
        n: Number of interpolation steps.
    """
    with torch.no_grad():
        # Encode to latent space
        z1 = model.encoder(x1.view(1, -1))
        z2 = model.encoder(x2.view(1, -1))

        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, n)
        interpolations = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])

        # Decode interpolations
        recon = model.decoder(interpolations).cpu().numpy()
        recon = recon.reshape(-1, 28, 28)  # Reshape to (n, 28, 28)

        # Normalize to [0, 255] and convert to uint8
        recon = (recon * 255).astype(np.uint8)

        # Concatenate images horizontally
        interpolation_strip = np.concatenate(recon, axis=1)

        # Resize for better visualization (optional)
        interpolation_strip = cv2.resize(
            interpolation_strip, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST
        )

        # Display
        cv2.imwrite('interpolate_' + tag + '.png', interpolation_strip)


if __name__ == "__main__":
    model = VAE(latent_dim=64)
    x = torch.randn(10, 3, 128, 128)
    xx, _, _ = model(x)
    print(x.shape, xx.shape)
