import os
import shutil


# Function to extract and evaluate models
def evaluate_models(mlruns_path,best_model_save_path):
    best_model = None
    best_mae = float('inf')
    best_rmse = float('inf')

    # Iterate through model directories
    for model_run in os.listdir(mlruns_path):
        model_run_path = os.path.join(mlruns_path, model_run)
        
        if os.path.isdir(model_run_path):
            metrics_path = os.path.join(model_run_path, 'metrics')

            # Check if metrics folder exists
            if os.path.exists(metrics_path):
                print(f"\nChecking metrics for model run: {model_run}")
                mae_value = None
                rmse_value = None

                for metric_file in os.listdir(metrics_path):
                    metric_file_path = os.path.join(metrics_path, metric_file)
                    
                    # Print all metric files
                    print(f"Found metric file: {metric_file}")
                    
                    try:
                        # Parse MAE and RMSE files and extract values
                        if 'MAE' in metric_file:
                            with open(metric_file_path, 'r') as f:
                                # Split the string by spaces and take the second value
                                mae_value_str = f.read().strip().split()[1]  # Assuming the MAE value is in the second position
                                mae_value = float(mae_value_str)
                        elif 'RMSE' in metric_file:
                            with open(metric_file_path, 'r') as f:
                                # Split the string by spaces and take the second value
                                rmse_value_str = f.read().strip().split()[1]  # Assuming the RMSE value is in the second position
                                rmse_value = float(rmse_value_str)
                    except Exception as e:
                        print(f"Error reading {metric_file}: {e}")

                # If both MAE and RMSE values are found, evaluate
                if mae_value is not None and rmse_value is not None:
                    print(f"MAE: {mae_value}, RMSE: {rmse_value}")

                    if mae_value < best_mae and rmse_value < best_rmse:
                        best_mae = mae_value
                        best_rmse = rmse_value
                        best_model = model_run

    # Report the best model and save it
    if best_model:
        print(f"\nBest Model: {best_model}")
        print(f"MAE: {best_mae}, RMSE: {best_rmse}")

        # Copy the best model to the specified save path
        best_model_artifact_path = os.path.join(mlruns_path, best_model, 'artifacts', 'pollution_lstm_model.h5')
        
        if os.path.exists(best_model_artifact_path):
            # Ensure the directory exists
            if not os.path.exists(best_model_save_path):
                os.makedirs(best_model_save_path)
            
            # Copy the model file to the save directory
            shutil.copy(best_model_artifact_path, os.path.join(best_model_save_path, 'best_pollution_lstm_model.h5'))
            print(f"Best model saved at: {best_model_save_path}best_pollution_lstm_model.h5")
        else:
            print(f"Model artifact not found in {best_model_artifact_path}")
    else:
        print("No valid models found with MAE and RMSE metrics.")

# Run the evaluation
#evaluate_models()
