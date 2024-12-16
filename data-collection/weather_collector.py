import os
import requests
import csv
import time
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
CITY = "Islamabad"
LATITUDE = 33.6844  # Islamabad's latitude
LONGITUDE = 73.0479  # Islamabad's longitude
INTERVAL = 3600  # Fetch data every 1 hour

# Ensure the `data` directory exists
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Parent directory
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, 'air_pollution_data.csv')

def dvc_add_data():
    """
    Add the updated CSV file to DVC tracking
    """
    try:
        subprocess.run(['dvc', 'add', OUTPUT_FILE], check=True)
        print(f"Added {OUTPUT_FILE} to DVC tracking")
    except subprocess.CalledProcessError as e:
        print(f"Error adding file to DVC: {e}")

def fetch_air_pollution_data(api_key, lat, lon):
    """
    Fetch air pollution data with error handling
    """
    query_url = f"{BASE_URL}?lat={lat}&lon={lon}&appid={api_key}"
    
    try:
        response = requests.get(query_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def write_to_csv(filename, data):
    """
    Write air pollution data to CSV file
    """
    file_exists = os.path.exists(filename)
    
    with open(filename, mode="a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write headers if file is newly created
        if not file_exists:
            headers = [
                "Timestamp", "City", "Longitude", "Latitude", "PM2.5 (µg/m³)", 
                "PM10 (µg/m³)", "CO (µg/m³)", "NO2 (µg/m³)", "O3 (µg/m³)"
            ]
            writer.writerow(headers)
        
        writer.writerow(data)

def main(max_iterations=24):
    """
    Main data collection loop with DVC integration
    """
    print(f"Starting air pollution data collection for {CITY}...")
    
    iterations = 0
    while iterations < max_iterations:
        try:
            pollution_data = fetch_air_pollution_data(API_KEY, LATITUDE, LONGITUDE)
            
            if pollution_data:
                # Extract relevant fields
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                components = pollution_data['list'][0]['components']
                pm2_5 = components.get('pm2_5', None)
                pm10 = components.get('pm10', None)
                co = components.get('co', None)
                no2 = components.get('no2', None)
                o3 = components.get('o3', None)

                # Prepare data row
                data_row = [timestamp, CITY, LONGITUDE, LATITUDE, pm2_5, pm10, co, no2, o3]
                print(f"Fetched data: {data_row}")
                
                # Write to CSV
                write_to_csv(OUTPUT_FILE, data_row)
                
                # Add to DVC tracking
                dvc_add_data()
            else:
                print("Failed to fetch data this iteration.")
            
            # Wait for the specified interval
            time.sleep(INTERVAL)
            iterations += 1
        
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    main()
