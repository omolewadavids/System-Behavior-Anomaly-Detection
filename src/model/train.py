import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.model.dataset import get_data_loader
from src.utils.config import *
from src.utils.logger import logger


# Define LSTM Model
class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]  # Take output from last time step
        output = self.fc(last_time_step)
        return self.sigmoid(output)  # Probability of being an anomaly


# Initialize model, loss function, and optimizer
device = torch.device(DEVICE)
model = LSTMAnomalyDetector(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for anomaly classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load dataset
train_loader = get_data_loader(
    batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES, anomaly_ratio=ANOMALY_RATIO
)

# Training loop
logger.info("🚀 Training started...")
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    model.train()

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Ensure labels are in the shape (batch_size, 1) for BCELoss
        labels = labels.squeeze()

        # Forward pass
        outputs = model(inputs).squeeze()  # Shape should be (batch_size,)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    logger.info(f"✅ Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
logger.info(f"🎉 Training complete! Model saved as '{MODEL_SAVE_PATH}'")
