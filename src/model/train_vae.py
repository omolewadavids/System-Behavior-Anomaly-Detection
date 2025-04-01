import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.model.vae_model import VAE
from src.model.dataset import generate_synthetic_data
from src.utils.vae_config import *
from src.utils.logger import logger

# Load data
data, _ = generate_synthetic_data(num_samples=NUM_SAMPLES, anomaly_ratio=ANOMALY_RATIO)
dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
device = torch.device(DEVICE)
vae = VAE(INPUT_SIZE, LATENT_DIM).to(device)
optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)

# Training Loop
logger.info("🚀 Training VAE started...")
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for batch in dataloader:
        x = batch[0].to(device)

        # Forward pass
        reconstructed, mu, logvar = vae(x)
        loss = vae.loss_function(reconstructed, x, mu, logvar)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"✅ Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f}")

# Save model
torch.save(vae.state_dict(), "vae_model.pth")
logger.info("🎉 VAE training complete! Model saved as 'vae_model.pth'.")
