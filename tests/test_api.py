from fastapi.testclient import TestClient
from src.api.main import app  # Import the FastAPI app
import json

client = TestClient(app)


def test_inference():
    # Sample input data (mock system metrics)
    data = {
        "cpu": [
            30.5,
            28.7,
            35.3,
            29.8,
            30.2,
        ],  # Example values for CPU usage (percentages)
        "memory": [
            60.0,
            59.5,
            58.9,
            60.2,
            59.7,
        ],  # Example values for memory usage (percentages)
        "disk": [
            80.0,
            75.2,
            76.5,
            79.3,
            81.2,
        ],  # Example values for disk usage (percentages)
        "network": [
            100.5,
            102.3,
            99.8,
            101.2,
            100.4,
        ],  # Example values for network usage (bytes/s)
        "temperature": [
            45.3,
            44.1,
            46.2,
            44.9,
            45.5,
        ],  # Example values for temperature (Celsius)
    }

    # Send POST request to inference endpoint
    response = client.post("/predict", json=data)

    # Check if the response is successful
    assert response.status_code == 200
    assert (
        "prediction" in response.json()
    )  # Ensure there's a prediction key in the response

    # Example of checking the prediction value (assuming it returns a binary value 0 or 1)
    prediction = response.json()["prediction"]
    assert prediction in [0, 1]  # Expected prediction should be either 0 or 1


def test_health():
    # Send GET request to health check endpoint
    response = client.get("/health")

    # Check if the response is successful
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}  # Example health status response
