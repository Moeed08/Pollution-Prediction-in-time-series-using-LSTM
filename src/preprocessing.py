import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):     
    """
    Comprehensive data preprocessing for time series pollution data.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        tuple: (scaled_data, column_names, scaler)
    """
    # Load data with explicit datetime parsing     
    data = pd.read_csv(file_path, parse_dates=['Timestamp'])
    
    # Set Timestamp as index
    data.set_index('Timestamp', inplace=True)
    
    # Separate non-numeric and numeric columns     
    non_numeric_columns = ['City']
    location_columns = ['Longitude', 'Latitude']
    numeric_columns = [col for col in data.columns if col not in non_numeric_columns + location_columns]
    
    # Create a copy of numeric data for processing
    numeric_data = data[numeric_columns].copy()
    
    # Advanced missing value handling
    # First, check for missing values
    print("Missing values before interpolation:")
    print(numeric_data.isnull().sum())
    
    # Time-based interpolation for missing values
    numeric_data.interpolate(method='time', inplace=True)
    
    print("Missing values after interpolation:")
    print(numeric_data.isnull().sum())
    
    # Robust outlier detection using Interquartile Range (IQR)
    def remove_outliers(df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Create a mask for rows without outliers
        mask = ~((df < lower_bound) | (df > upper_bound)).any(axis=1)
        return df[mask]
    
    # Remove outliers
    numeric_data = remove_outliers(numeric_data)
    
    # Normalize numeric data     
    scaler = MinMaxScaler()     
    scaled_data = scaler.fit_transform(numeric_data)
    
    # Debugging information
    print("Original data shape:", data.shape)
    print("Processed numeric data shape:", numeric_data.shape)
    print("Scaled data shape:", scaled_data.shape)
    
    return scaled_data, numeric_data.columns, scaler 

# Optional: Add logging or more detailed preprocessing insights
def get_preprocessing_insights(original_data, processed_data):
    """
    Generate insights about the preprocessing step.
    """
    insights = {
        "total_rows_original": len(original_data),
        "total_rows_after_preprocessing": len(processed_data),
        "rows_removed": len(original_data) - len(processed_data),
        "removal_percentage": (len(original_data) - len(processed_data)) / len(original_data) * 100
    }
    return insights