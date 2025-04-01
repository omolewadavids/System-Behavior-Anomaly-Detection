import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_size=5, latent_dim=2):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Latent space
        self.mu_layer = nn.Linear(16, latent_dim)  # Mean
        self.logvar_layer = nn.Linear(16, latent_dim)  # Log variance

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),  # Output should match input size
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from the latent distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)
        mu, logvar = self.mu_layer(encoded), self.logvar_layer(encoded)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decoding
        reconstructed = self.decoder(z)

        return reconstructed, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """Computes VAE loss: reconstruction + KL divergence"""
        recon_loss = nn.MSELoss()(recon_x, x)  # Reconstruction loss
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_divergence
