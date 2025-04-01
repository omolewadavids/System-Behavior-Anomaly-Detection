import torch

# General settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# VAE Model Parameters
INPUT_SIZE = 5  # Number of system metrics (CPU, memory, disk, network, temperature)
LATENT_DIM = 2  # Size of the latent space (compressing the input features)
HIDDEN_SIZE = 16  # Hidden layer size
NUM_LAYERS = 1  # Number of LSTM layers (not used in VAE)
OUTPUT_SIZE = INPUT_SIZE  # VAE reconstructs the same input features
ANOMALY_RATIO = 0.1  # Fraction of anomalies in dataset
NUM_SAMPLES = 2000  # Number of samples for training

# Training settings
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Anomaly detection threshold (reconstruction error threshold)
ANOMALY_THRESHOLD = 0.02

# File paths
MODEL_SAVE_PATH = "vae_model.pth"
