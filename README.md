# System Anomaly Detection Model

This repository contains a **System Anomaly Detection Model** using **PyTorch LSTM** for detecting anomalies in system metrics like CPU, memory, disk, network usage, and temperature. The project includes:
- **Data Generation**: Creating synthetic data to simulate system metrics and anomalies.
- **Data Preprocessing**: Processing and preparing the data for model training.
- **Model Training**: Building and training a model to detect system anomalies.
- **Model Deployment**: Deploying the trained model using Docker and FastAPI.
- **CI/CD Pipeline**: Automating the deployment and model updates via GitHub Actions.

---

## **Project Structure**

```plaintext
system-anomaly-detection/
├── app/
│   ├── Dockerfile                # Dockerfile for the FastAPI app
│   ├── entrypoint.sh             # Shell script to run FastAPI app
│   ├── main.py                   # FastAPI app
│   ├── model.py                  # System anomaly detection model
│   ├── preprocess.py             # Data preprocessing code
│   ├── requirements.txt          # Python dependencies for the FastAPI app
│   └── utils.py                  # Utility functions for preprocessing and anomaly detection
├── data/
│   ├── generate_synthetic_data.py # Script to generate synthetic system data
├── infrastructure/
│   ├── main.tf                   # Terraform configuration for AWS resources
│   ├── variables.tf              # Terraform variables
│   ├── outputs.tf                # Terraform outputs
│   └── provider.tf               # AWS provider setup for Terraform
└── .github/
    └── workflows/
        └── deploy.yml            # CI/CD GitHub Actions workflow
