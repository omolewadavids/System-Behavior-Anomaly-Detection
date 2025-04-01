import torch
import numpy as np
from src.model.train import LSTMAnomalyDetector


def load_model():
    model = LSTMAnomalyDetector(input_size=10, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    return model


def predict_anomaly(model, data):
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(data_tensor)
    return output.item()
