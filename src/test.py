import numpy as np
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error,
    r2_score,
    confusion_matrix,
    classification_report
)

from src.train import create_sequences

def prepare_test_data(scaled_data, sequence_length, target_column=0):
    """
    Prepare test data for model evaluation.
    
    Args:
    - scaled_data: Preprocessed scaled data
    - sequence_length: Length of input sequences
    - target_column: Index of column to predict
    
    Returns:
    - X_test, y_test for model evaluation
    """
    try:
        X_test, y_test = create_sequences(scaled_data, sequence_length, target_column)
        
        # Ensure X_test has 3 dimensions (samples, timesteps, features)
        if len(X_test.shape) < 3:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        return X_test, y_test
    except Exception as e:
        print(f"Error preparing test data: {e}")
        return None, None

def test_model(model, scaled_data, sequence_length, scaler, target_column=0):
    """
    Comprehensive model testing function.
    
    Args:
    - model: Trained LSTM model
    - scaled_data: Preprocessed scaled data
    - sequence_length: Length of input sequences
    - scaler: Data scaler for inverse transformation
    - target_column: Index of column to predict
    
    Returns:
    - Dictionary of test results and metrics
    """
    # Start MLflow run for testing
    with mlflow.start_run(run_name="Model Testing", nested=True):
        try:
            # Prepare test data
            X_test, y_test = prepare_test_data(scaled_data, sequence_length, target_column)
            
            if X_test is None or y_test is None:
                print("Could not prepare test data")
                return None

            # Predict
            y_pred = model.predict(X_test).flatten()
            y_test = y_test.flatten()

            # Compute metrics
            metrics = {
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAE": mean_absolute_error(y_test, y_pred),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred)
            }

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)

            # Visualization: Actual vs Predicted
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.7)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.title('Actual vs Predicted Values')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.tight_layout()
            plt.savefig('actual_vs_predicted.png')
            plt.close()

            # Residual Plot
            residuals = y_test - y_pred
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.7)
            plt.title('Residual Plot')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.tight_layout()
            plt.savefig('residual_plot.png')
            plt.close()

            # Print and log detailed results
            print("Model Testing Results:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
                mlflow.log_metric(metric, value)

            # Optional: Attach plots to MLflow
            mlflow.log_artifact('actual_vs_predicted.png')
            mlflow.log_artifact('residual_plot.png')

            return {
                "metrics": metrics,
                "y_true": y_test,
                "y_pred": y_pred
            }

        except Exception as e:
            print(f"Error during model testing: {e}")
            mlflow.set_tag("error", str(e))
            return None