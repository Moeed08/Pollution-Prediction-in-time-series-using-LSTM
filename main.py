from src.preprocessing import preprocess_data
from src.train import train_lstm
from src.test import test_model
import pandas as pd
from bestmodel.bestmodel import evaluate_models
if __name__ == "__main__":
    # Configuration
    FILE_PATH = 'data/air_pollution_data.csv'
    mlruns_path = 'E:/mlops-proj-copy/mlruns/0/'
    best_model_save_path = 'E:/mlops-proj-copy/bestmodel/'
    
    # Read data to determine sequence length dynamically
    data = pd.read_csv(FILE_PATH)
    SEQUENCE_LENGTH = min(5, len(data) - 1)  # Dynamically adjust sequence length
    target_column = 0  # Usually first pollution metric

    # Step 1: Preprocess data
    scaled_data, columns, scaler = preprocess_data(FILE_PATH)

    # Step 2: Train LSTM model
    lstm_model = train_lstm(scaled_data, SEQUENCE_LENGTH, scaler, target_column)

    # Step 3: Test the model
    test_results = test_model(lstm_model, scaled_data, SEQUENCE_LENGTH, scaler, target_column)

    bestModel=evaluate_models(mlruns_path,best_model_save_path)


    