import os
import torch

# ----------------------------
# General Settings
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42  # Random seed for reproducibility

# ----------------------------
# Model Hyperparameters
# ----------------------------
INPUT_SIZE = 5  # Number of system metrics (CPU, memory, etc.)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1

# ----------------------------
# Training Settings
# ----------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
ANOMALY_RATIO = 0.1  # Fraction of anomalies in dataset
NUM_SAMPLES = 2000  # Number of samples for training

# ----------------------------
# File Paths
# ----------------------------
MODEL_SAVE_PATH = os.path.join(os.getcwd(), "model.pth")
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)



