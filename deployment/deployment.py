import os
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Initialize Flask app
app = Flask(__name__)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Application info', version='1.0.0')

# Custom Prometheus metrics
data_ingestion_counter = Counter('data_ingestion_requests', 'Number of data ingestion requests')
prediction_counter = Counter('model_predictions', 'Number of model predictions made')
response_time_histogram = Histogram('api_response_time_seconds', 'API response time in seconds')

# Path to the model inside the container
model_path = os.path.join("E:/mlops-proj-copy/pollution_lstm_model.h5")

# Custom model loader with error handling
def load_custom_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path, 
            custom_objects={
                'time_major': False  # Remove or handle the time_major parameter
            }, 
            compile=False  # Prevent recompilation
        )
        print(f"Model successfully loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Load the model
try:
    model = load_custom_model(model_path)
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

# Add this specific route for Prometheus to scrape metrics
@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/')
def home():
    return "Welcome to the Pollution Prediction Flask App Use 1) /predict for predictions 2) /health for the status."

@app.route('/health', methods=['GET'])
def health():
    if model is not None:
        return jsonify({'status': 'API is running', 'model_loaded': True})
    else:
        return jsonify({'status': 'API is running', 'model_loaded': False}), 500

@app.before_request
def track_request():
    data_ingestion_counter.inc()

@app.route('/predict', methods=['POST'])
@response_time_histogram.time()
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500

    try:
        # Get input data from JSON request
        data = request.get_json()

        # Ensure the features are in the correct shape
        features = np.array(data['features'])

        # Validate features length (it should be 6)
        if len(features) != 6:
            return jsonify({'status': 'error', 'message': 'Input must contain 6 features'})

        # Replicate the features across 10 time steps to match the model's expected input shape
        features = np.tile(features, (10, 1))  # Replicate across 10 time steps

        # Reshape to (1, 10, 6) as expected by the LSTM model
        features = np.reshape(features, (1, 10, 6))

        # Make prediction
        prediction = model.predict(features)
        prediction_counter.inc()

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on all available interfaces
    app.run(debug=True, host='0.0.0.0', port=5000)
