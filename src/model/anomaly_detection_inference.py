import torch
import numpy as np
from src.model.vae_model import VAE
from src.utils.config import *

# Load trained model
device = torch.device(DEVICE)
vae = VAE(INPUT_SIZE, LATENT_DIM).to(device)
vae.load_state_dict(torch.load("vae_model.pth"))
vae.eval()


def detect_anomalies(data, threshold=0.02):
    """Detect anomalies based on reconstruction error."""
    data = torch.tensor(data, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed, _, _ = vae(data)

    # Compute reconstruction error (Mean Squared Error)
    errors = ((data - reconstructed) ** 2).mean(dim=1).cpu().numpy()

    # Mark samples with high error as anomalies
    anomalies = errors > threshold
    return anomalies, errors


# Example usage
sample_data = np.array([[30.5, 60.0, 80.0, 100.5, 45.3]])
anomalies, errors = detect_anomalies(sample_data)
print(f"Anomaly Detected: {anomalies[0]}, Reconstruction Error: {errors[0]}")
