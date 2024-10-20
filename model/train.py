import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump
from torch.optim.lr_scheduler import ReduceLROnPlateau
from imblearn.over_sampling import SMOTE


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


class EsiModel(nn.Module):
    def __init__(self, input_size):
        super(EsiModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 5)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
        x = self.dropout(torch.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        return x


def create_features(df):
    # Check for 'age' column and create age groups if it exists
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"], bins=[0, 18, 30, 50, 70, 100], labels=[0, 1, 2, 3, 4]
        )

    # Check for 'weight' and 'height' columns before calculating BMI
    if "weight" in df.columns and "height" in df.columns:
        df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

    # Add more feature engineering here based on available columns
    # For example:
    if "arrival_mode" in df.columns:
        df["is_ambulance"] = (df["arrival_mode"] == "ambulance").astype(int)

    if "chief_complaint" in df.columns:
        # Create dummy variables for chief complaints
        chief_complaint_dummies = pd.get_dummies(df["chief_complaint"], prefix="cc")
        df = pd.concat([df, chief_complaint_dummies], axis=1)

    return df


def train_esi_model(
    df, target_column="esi", num_epochs=10, batch_size=64, learning_rate=0.001
):
    print("Original columns:", df.columns)
    df = create_features(df)
    print("Columns after feature engineering:", df.columns)

    X = df.drop(columns=[target_column])
    y = df[target_column] - 1  # Adjust labels to be 0-indexed

    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Compute class weights
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_resampled), y=y_resampled
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_accuracy = 0
    best_model = None

    for fold, (train_index, val_index) in enumerate(kf.split(X_resampled), 1):
        print(f"Fold {fold}")

        X_train, X_val = X_resampled.iloc[train_index], X_resampled.iloc[val_index]
        y_train, y_val = y_resampled.iloc[train_index], y_resampled.iloc[val_index]

        train_dataset = CustomDataset(X_train, y_train)
        val_dataset = CustomDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = EsiModel(X.shape[1])
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5, verbose=True
        )

        for epoch in range(num_epochs):
            model.train()
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    outputs = model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.numpy())
                    val_labels.extend(batch_labels.numpy())

            val_accuracy = accuracy_score(val_labels, val_preds)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}"
            )

            scheduler.step(val_accuracy)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model.state_dict()

        print(f"Best Validation Accuracy: {best_accuracy:.4f}")

    # Load the best model
    final_model = EsiModel(X.shape[1])
    final_model.load_state_dict(best_model)

    return final_model, scaler, X.columns


# Load and prepare the data
df = pd.read_csv("data/processed/triage_cleaned.csv")
print("Columns in the dataset:", df.columns)

if "esi" not in df.columns:
    raise ValueError("The 'esi' column is not present in the dataset.")

# Train the model
model, scaler, feature_columns = train_esi_model(df)

# Save the model, scaler, and feature columns
torch.save(model.state_dict(), "esi_model.pth")
dump(scaler, "scaler.joblib")
dump(feature_columns, "feature_columns.joblib")

print("Model, scaler, and feature columns saved successfully!")

# Final evaluation on a held-out test set
X = df.drop(columns=["esi"])
y = df["esi"] - 1
X = create_features(X)
X = pd.get_dummies(X, drop_first=True)
X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        outputs = model(batch_features)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Final Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(
    classification_report(
        all_labels, all_preds, target_names=[f"ESI {i}" for i in range(1, 6)]
    )
)
