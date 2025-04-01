import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AnomalyDataset(Dataset):
    """
    Custom PyTorch Dataset for system anomaly detection.
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def generate_synthetic_data(num_samples=1000, time_steps=20, anomaly_ratio=0.1):
    """
    Generates synthetic time-series data simulating system performance metrics with anomalies.

    Parameters:
    - num_samples: Total number of sequences
    - time_steps: Number of time steps per sequence
    - anomaly_ratio: Fraction of sequences that are anomalous

    Returns:
    - data: NumPy array of shape (num_samples, time_steps, 5) for 5 system metrics
    - labels: NumPy array of shape (num_samples, 1) where 1 indicates an anomaly
    """
    # Normal operating ranges
    normal_cpu = np.random.normal(
        loc=50, scale=10, size=(num_samples, time_steps)
    )  # CPU usage (50% avg, ±10%)
    normal_memory = np.random.normal(
        loc=60, scale=5, size=(num_samples, time_steps)
    )  # Memory usage (60% avg, ±5%)
    normal_disk = np.random.normal(
        loc=40, scale=8, size=(num_samples, time_steps)
    )  # Disk usage (40% avg, ±8%)
    normal_network = np.random.normal(
        loc=100, scale=20, size=(num_samples, time_steps)
    )  # Network traffic (100 Mbps avg, ±20)
    normal_temperature = np.random.normal(
        loc=70, scale=5, size=(num_samples, time_steps)
    )  # Temperature (70°C avg, ±5°C)

    # Stack into a single dataset (num_samples, time_steps, 5 features)
    data = np.stack(
        [normal_cpu, normal_memory, normal_disk, normal_network, normal_temperature],
        axis=-1,
    )
    labels = np.zeros((num_samples, 1))  # Default: normal (label = 0)

    # Inject anomalies
    num_anomalies = int(num_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(
            ["cpu_spike", "memory_leak", "disk_failure", "network_spike", "overheating"]
        )
        time_idx = np.random.randint(
            0, time_steps
        )  # Random time step for anomaly injection

        if anomaly_type == "cpu_spike":
            data[idx, time_idx:, 0] += np.random.uniform(20, 50)  # Sudden CPU spike
        elif anomaly_type == "memory_leak":
            data[idx, time_idx:, 1] += np.linspace(
                0, 30, num=time_steps - time_idx
            )  # Gradual memory increase
        elif anomaly_type == "disk_failure":
            data[idx, time_idx:, 2] = np.random.uniform(80, 100)  # Sudden disk overload
        elif anomaly_type == "network_spike":
            data[idx, time_idx:, 3] += np.random.uniform(
                50, 100
            )  # Unusual network spike
        elif anomaly_type == "overheating":
            data[idx, time_idx:, 4] += np.random.uniform(10, 30)  # System overheating

        labels[idx] = 1  # Mark as anomaly

    return data, labels


def get_data_loader(batch_size=32, num_samples=1000, anomaly_ratio=0.1):
    """
    Returns a DataLoader for training.
    """
    data, labels = generate_synthetic_data(
        num_samples=num_samples, anomaly_ratio=anomaly_ratio
    )
    dataset = AnomalyDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    # Test data loading
    loader = get_data_loader()
    for batch in loader:
        x, y = batch
        print(f"Batch X Shape: {x.shape}, Batch Y Shape: {y.shape}")
        print(f"Sample Labels: {y[:10].squeeze().tolist()}")  # Show first few labels
        break
