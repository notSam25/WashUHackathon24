import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler


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


def load_model():
    model_state_dict = torch.load(
        "api/model/saved_models/esi_model.pth", weights_only=True
    )
    scaler = load("api/model/saved_models/scaler.joblib")
    feature_columns = load("api/model/saved_models/feature_columns.joblib")

    model = EsiModel(len(feature_columns))
    model.load_state_dict(model_state_dict)
    model.eval()

    return model, scaler, feature_columns


def create_features(df):
    # Feature engineering logic matching the training script
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"], bins=[0, 18, 30, 50, 70, 100], labels=[0, 1, 2, 3, 4]
        )

    # Additional feature engineering based on available columns
    return df


def preprocess_data(df, scaler, feature_columns):
    print(df)
    df = pd.get_dummies(df, drop_first=True)
    df = create_features(df)

    # Ensure all columns from training are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training data
    df = df[feature_columns]

    # Scale the input data
    scaled_input = scaler.transform(df)

    return scaled_input


def predict_esi(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    # Convert to ESI score (1-indexed)
    esi_scores = predicted_class.numpy() + 1
    return esi_scores, probabilities.numpy()


def predict_from_dataframe(input_data):
    # Load model, scaler, and feature columns
    model, scaler, feature_columns = load_model()
    print("Hello")

    # Preprocess the input data
    scaled_input = preprocess_data(input_data, scaler, feature_columns)
    print("Hello2")

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    print("Hello3")

    # Make predictions
    esi_scores, probabilities = predict_esi(model, input_tensor)
    print("Hello4")

    # Return the predictions and probabilities
    return esi_scores, probabilities


def predict():
    input_csv_path = "data/test/test.csv"  # Replace this with your input CSV path
    input_data = pd.read_csv(input_csv_path)

    # Get ESI predictions and probabilities
    esi_scores, probabilities = predict_from_dataframe(input_data)

    # Print the predictions
    for i, score in enumerate(esi_scores):
        print(
            f"Sample {i + 1}: Predicted ESI = {score}, Probabilities = {probabilities[i]}"
        )
