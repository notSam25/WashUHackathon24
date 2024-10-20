import pandas as pd
import torch
from torch import nn
from joblib import load


# Neural Network Model
class EsiModel(nn.Module):
    def __init__(self, input_size):
        super(EsiModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 5)  # 5 output classes for ESI 1-5
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Load the test data
test_data = pd.read_csv("data/test/test.csv")

# Ensure the test data has the same columns as the training data (excluding 'esi')
# You may need to adjust this based on your actual data
if "esi" in test_data.columns:
    test_data = test_data.drop(columns=["esi"])

# Load the scaler
scaler = load("scaler.joblib")

# Preprocess the test data
test_data_scaled = scaler.transform(test_data)

# Load model
input_size = test_data.shape[1]
model = EsiModel(input_size)
model.load_state_dict(torch.load("esi_model.pth"))
model.eval()

# Convert to PyTorch tensor
test_tensor = torch.FloatTensor(test_data_scaled)

# Make predictions
with torch.no_grad():
    output = model(test_tensor)
    predicted_classes = torch.argmax(output, dim=1).numpy()

# Add 1 to predictions to match original ESI scale (1-5)
predicted_esi = predicted_classes + 1

# Add predictions to the test data
test_data["Predicted_ESI"] = predicted_esi

# Save results
test_data.to_csv("test_results.csv", index=False)

print("Predictions complete. Results saved to 'test_results.csv'.")
