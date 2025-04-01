# System Anomaly Detection Model

This repository contains a **System Anomaly Detection Model** using **PyTorch LSTM** for detecting anomalies in system metrics like CPU, memory, disk, network usage, and temperature. The project includes:
- **Data Generation**: Creating synthetic data to simulate system metrics and anomalies.
- **Data Preprocessing**: Processing and preparing the data for model training.
- **Model Training**: Building and training a model to detect system anomalies.
- **Model Deployment**: Deploying the trained model using Docker and FastAPI.
- **CI/CD Pipeline**: Automating the deployment and model updates via GitHub Actions.

---

## **Table of Contents**

1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Generate Synthetic Data](#generate-synthetic-data)
4. [Preprocess Data](#preprocess-data)
5. [Model Training](#model-training)
6. [Docker Deployment](#docker-deployment)
7. [Terraform AWS Infrastructure](#terraform-aws-infrastructure)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [How to Use](#how-to-use)
10. [License](#license)
11. [Acknowledgments](#acknowledgments)

---

## **Requirements**

### Software Dependencies:
- **Python 3.x**
- **FastAPI** for building the API
- **PyTorch** for model building and training
- **Docker** for containerization
- **Terraform** for AWS infrastructure management
- **AWS CLI** for interacting with AWS services
- **GitHub Actions** for automating deployment

### Python Libraries:
- Install required Python dependencies by running:

  ```bash
  pip install -r app/requirements.txt


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

