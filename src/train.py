import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.keras

def create_lstm_model(input_shape):
    """
    Build an LSTM model for time-series prediction.
    """
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Add metrics for better tracking
    return model

def create_sequences(data, sequence_length, target_column_index=0):
   
    if len(data) < sequence_length:
        raise ValueError(f"Not enough data. Need at least {sequence_length} rows, got {len(data)}")
    
    X, y = [], []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length, :]
        X.append(seq)
        y.append(data[i + sequence_length, target_column_index])

    return np.array(X), np.array(y)

def train_lstm(data, sequence_length, scaler, target_column_index=0):
    """
    Train LSTM and log experiments with MLflow.
    """
    mlflow.set_experiment("Pollution Prediction LSTM")
    mlflow.keras.autolog()

    if not isinstance(data, np.ndarray):
        data = data.to_numpy()

    print(f"Input data shape: {data.shape}")
    
    try:
        X, y = create_sequences(data, sequence_length, target_column_index)
    except ValueError as e:
        print(f"Error creating sequences: {e}")
        sequence_length = max(1, len(data) // 2)
        X, y = create_sequences(data, sequence_length, target_column_index)

    print(f"Sequences X shape: {X.shape}")
    print(f"Sequences y shape: {y.shape}")

    if len(X.shape) < 3:
        X = X.reshape(X.shape[0], X.shape[1], 1)

    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        print("Not enough data for train-test split. Using entire dataset.")
        X_train, X_test = X, X
        y_train, y_test = y, y

    with mlflow.start_run():
        try:
            model = create_lstm_model((X.shape[1], X.shape[2]))
            epochs = min(50, max(10, len(X) * 2))
            
            history = model.fit(
                X_train, y_train, 
                validation_data=(X_test, y_test) if len(X_test) > 0 else None, 
                epochs=epochs, 
                batch_size=min(32, len(X_train)),
                verbose=1
            )

            if len(X_test) > 0:
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)

                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE", mae)
            else:
                print("Warning: No test data available for evaluation")

            # Save the model locally
            model_save_path = "pollution_lstm_model.h5"
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")

            # Log the saved model to MLflow artifacts
            mlflow.log_artifact(model_save_path)

        except Exception as e:
            print(f"Training failed: {e}")
            raise

    return model

def make_prediction(model, input_sequence, scaler=None):
    """
    Make a prediction using the trained model.
    """
    prediction = model.predict(input_sequence)
    if scaler:
        prediction = scaler.inverse_transform(prediction)
    return prediction



