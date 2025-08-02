import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class AQIDataFetcher:
    def __init__(self):
        # Using free APIs - replace with actual keys
        self.aqi_api_key = "d83ef1d2fb4bbfee819111089bcef322f84ce471"  # Replace with actual AQICN API key
        self.weather_api_key = "e9b863901ee71e7b2e720bf36e69ca87"  # Replace with actual OpenWeather API key
        self.lahore_coords = {"lat": 31.5204, "lon": 74.3587}
        
    def fetch_historical_aqi_data(self, start_date, end_date):
        """Fetch historical AQI data for Lahore"""
        data = []
        current_date = start_date
        
        while current_date <= end_date:
            # Simulate AQI data (replace with actual API call)
            aqi_value = np.random.randint(50, 300)  # Realistic AQI range for Lahore
            pm25 = np.random.uniform(20, 150)
            pm10 = np.random.uniform(30, 200)
            
            data.append({
                'timestamp': current_date,
                'aqi': aqi_value,
                'pm25': pm25,
                'pm10': pm10,
                'city': 'Lahore',
                'country': 'Pakistan'
            })
            
            current_date += timedelta(hours=1)
            
        return pd.DataFrame(data)
    
    def fetch_current_aqi_data(self):
        """Fetch current AQI data from AQICN API"""
        city = "lahore"
        url = f"https://api.waqi.info/feed/{city}/?token={self.aqi_api_key}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "ok":
                aqi = data["data"]["aqi"]
                iaqi = data["data"]["iaqi"]
                pm25 = iaqi.get("pm25", {}).get("v", None)
                pm10 = iaqi.get("pm10", {}).get("v", None)
                
                df = pd.DataFrame([{
                    'timestamp': datetime.now(),
                    'aqi': aqi,
                    'pm25': pm25,
                    'pm10': pm10,
                    'city': 'Lahore',
                    'country': 'Pakistan'
                }])
                return df
            else:
                print("Error in AQICN API response:", data)
        else:
            print("Failed to fetch AQI data. Status code:", response.status_code)
        return pd.DataFrame()

    def fetch_weather_data(self, start_date=None, end_date=None):
        """Fetch weather data:
        - If start_date & end_date are provided → simulate historical data
        - Else → fetch current weather from OpenWeather API
        """
        if start_date and end_date:
            # Simulate Historical Weather Data
            data = []
            current_date = start_date
            while current_date <= end_date:
                temp = np.random.uniform(25, 45)
                humidity = np.random.uniform(30, 80)
                wind_speed = np.random.uniform(2, 15)
                pressure = np.random.uniform(1010, 1020)

                data.append({
                    'timestamp': current_date,
                    'temperature': temp,
                    'humidity': humidity,
                    'pressure': pressure,
                    'wind_speed': wind_speed
                })
                current_date += timedelta(hours=1)

            return pd.DataFrame(data)

        else:
            # Fetch Current Weather from OpenWeather API
            lat, lon = self.lahore_coords["lat"], self.lahore_coords["lon"]
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                main = data["main"]
                wind = data["wind"]

                df = pd.DataFrame([{
                    'timestamp': datetime.now(),
                    'temperature': main.get('temp'),
                    'humidity': main.get('humidity'),
                    'pressure': main.get('pressure'),
                    'wind_speed': wind.get('speed')
                }])
                return df
            else:
                print("Failed to fetch weather data. Status code:", response.status_code)
                return pd.DataFrame()
    
    def save_data_to_csv(self, data, filename):
        """Save data to CSV file in absolute path"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to project root
        raw_data_dir = os.path.join(base_dir, 'data', 'raw')
        
        # Ensure the directory exists
        os.makedirs(raw_data_dir, exist_ok=True)
        
        filepath = os.path.join(raw_data_dir, filename)
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

        
    def run_backfill(self, start_date_str="2024-07-01"):
        """Run backfill process from July 1st to today"""
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.now()

        print(f"Starting backfill from {start_date} to {end_date}")

        # Fetch AQI data (simulated)
        aqi_data = self.fetch_historical_aqi_data(start_date, end_date)
        self.save_data_to_csv(aqi_data, f"aqi_data_backfill_{start_date.strftime('%Y%m%d')}.csv")

        # Fetch Weather data (simulated)
        weather_data = self.fetch_weather_data(start_date, end_date)
        self.save_data_to_csv(weather_data, f"weather_data_backfill_{start_date.strftime('%Y%m%d')}.csv")

        print("Backfill completed successfully!")

        
    def run_live_fetch(self):
        """Fetch current data for live mode"""
        print("Fetching current data...")
        
        # Fetch current AQI
        current_aqi = self.fetch_current_aqi_data()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.save_data_to_csv(current_aqi, f"aqi_data_live_{timestamp}.csv")
        
        # Fetch current weather
        current_weather = self.fetch_weather_data()
        self.save_data_to_csv(current_weather, f"weather_data_live_{timestamp}.csv")
        
        print("Live data fetch completed!")

if __name__ == "__main__":
    fetcher = AQIDataFetcher()
    
    # Run backfill for historical data
    fetcher.run_backfill()
    
    # Fetch current data
    fetcher.run_live_fetch()
