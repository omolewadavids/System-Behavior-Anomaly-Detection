from fastapi import APIRouter, Depends
import numpy as np
from src.model.inference import predict_anomaly
from src.api.dependencies import get_model

from src.model.vae_model import VAE
from src.utils.config import *


device = torch.device(DEVICE)
vae = VAE(INPUT_SIZE, LATENT_DIM).to(device)
vae.load_state_dict(torch.load("vae_model.pth"))
vae.eval()

router = APIRouter()


@router.post("/predict/")
def predict(data: list, model=Depends(get_model)):
    prediction = predict_anomaly(model, np.array(data))
    return {"anomaly_score": prediction}


# Load VAE model
@router.post("/detect_anomaly")
async def detect_anomaly(data: dict):
    """API endpoint for VAE anomaly detection"""
    input_data = np.array(
        [
            data["cpu"],
            data["memory"],
            data["disk"],
            data["network"],
            data["temperature"],
        ]
    )
    input_data = input_data.reshape(1, -1)  # Reshape to match VAE input
    input_data = torch.tensor(input_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed, _, _ = vae(input_data)

    # Compute reconstruction error
    error = ((input_data - reconstructed) ** 2).mean().item()
    anomaly = error > 0.02  # Define a threshold for anomaly detection

    return {"anomaly_detected": anomaly, "reconstruction_error": error}
