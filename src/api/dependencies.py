from fastapi import Depends
import torch
from src.model.inference import load_model

# Dependency function to load the model once and reuse it
def get_model():
    model = load_model()  # Load the trained LSTM model
    return model
