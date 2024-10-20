import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(
            features.values if isinstance(features, pd.DataFrame) else features,
            dtype=torch.float32,
        )
        self.labels = torch.tensor(
            labels.values if isinstance(labels, pd.Series) else labels, dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Neural Network Model
class EsiModel(nn.Module):
    def __init__(self, input_size):
        super(EsiModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 5)  # 5 output classes for ESI 1-5
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# Main training function
def train_esi_model(
    df, target_column="esi", num_epochs=50, batch_size=128, learning_rate=0.001
):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column] - 1  # Adjust labels to be 0-indexed

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Preprocess the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    input_size = X.shape[1]
    model = EsiModel(input_size)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(batch_labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels, all_preds, target_names=[f"ESI {i}" for i in range(1, 6)]
        )
    )

    return model, scaler, X.columns


# Load and prepare the data
df = pd.read_csv(
    "data/processed/triage_cleaned.csv"
)  # Replace with your actual dataset file name

# Ensure 'esi' is the target column and it's present in the dataset
if "esi" not in df.columns:
    raise ValueError("The 'esi' column is not present in the dataset.")

# Train the model
model, scaler, feature_columns = train_esi_model(df)

# Save the model, scaler, and feature columns
torch.save(model.state_dict(), "esi_model.pth")
dump(scaler, "scaler.joblib")
dump(feature_columns, "feature_columns.joblib")

print("Model, scaler, and feature columns saved successfully!")
