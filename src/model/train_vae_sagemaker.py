import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import boto3
import numpy as np

from src.model.vae_model import VAE
from src.model.dataset import generate_synthetic_data
from src.utils.vae_config import *

# Parse SageMaker arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
parser.add_argument("--s3-bucket", type=str, required=True, help="S3 bucket to save model")
args = parser.parse_args()

# Load synthetic data
data, _ = generate_synthetic_data(num_samples=NUM_SAMPLES, anomaly_ratio=ANOMALY_RATIO)
dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Initialize model
device = torch.device(DEVICE)
vae = VAE(INPUT_SIZE, LATENT_DIM).to(device)
optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

# Start MLflow experiment
mlflow.set_experiment("VAE_Anomaly_Detection")
with mlflow.start_run():
    mlflow.log_params({"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate})

    print("🚀 Training started on AWS SageMaker...")
    for epoch in range(args.epochs):
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
        mlflow.log_metric("loss", avg_loss, step=epoch)
        print(f"✅ Epoch [{epoch + 1}/{args.epochs}] - Loss: {avg_loss:.4f}")

    # Save model
    model_path = "/opt/ml/model/vae_model.pth"
    torch.save(vae.state_dict(), model_path)
    print("🎉 Model training complete. Uploading to S3...")

    # Upload model to S3
    s3_client = boto3.client("s3")
    s3_client.upload_file(model_path, args.s3_bucket, "vae_model.pth")
    print(f"✅ Model saved in S3: s3://{args.s3_bucket}/vae_model.pth")

    # Log model to MLflow
    mlflow.pytorch.log_model(vae, "vae_model")
