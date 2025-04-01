import torch
from src.model.train import LSTMAnomalyDetector  # Assuming the model is defined here
from src.utils.config import *  # Import configuration for model setup
from src.model.dataset import get_data_loader  # Import dataset loader


# Load the trained model
def load_model():
    model = LSTMAnomalyDetector(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))  # Load saved model weights
    model.eval()  # Set model to evaluation mode
    return model


# Test model inference
def test_model_inference():
    # Load the trained model
    model = load_model()

    # Sample test data (mock system metrics)
    test_data = torch.tensor([[
        [30.5, 60.0, 80.0, 100.5, 45.3]  # Single data sample with CPU, memory, disk, network, temperature
    ]])

    # Move to device (GPU or CPU)
    test_data = test_data.float().to(DEVICE)

    # Make prediction
    with torch.no_grad():
        output = model(test_data).squeeze()

    # Check the output shape and value (should be between 0 and 1 for binary classification)
    assert output.shape == torch.Size([1])
    assert 0 <= output.item() <= 1  # Output should be a probability in the range [0, 1]

    # Example: Check if output is a reasonable anomaly detection score
    print(f"Prediction probability: {output.item()}")
    assert output.item() <= 1.0  # Probability cannot be greater than 1

