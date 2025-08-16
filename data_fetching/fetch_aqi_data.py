# #fetch_aqi_dat.py
# import requests
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import os
# import json

# class AQIDataFetcher:
#     def __init__(self):
#         # Using free APIs - replace with actual keys
#         self.aqi_api_key = "03f59b0cc6f536e1645c54fe248c0a7d5d9f04a2"  # Replace with actual AQICN API key
#         self.weather_api_key = "e9b863901ee71e7b2e720bf36e69ca87"  # Replace with actual OpenWeather API key
#         self.lahore_coords = {"lat": 31.5204, "lon": 74.3587}
        
#     def fetch_historical_aqi_data(self, start_date, end_date):
#         """Fetch historical AQI data for Lahore"""
#         data = []
#         current_date = start_date
        
#         while current_date <= end_date:
#             # Simulate AQI data (replace with actual API call)
#             aqi_value = np.random.randint(50, 300)  # Realistic AQI range for Lahore
#             pm25 = np.random.uniform(20, 150)
#             pm10 = np.random.uniform(30, 200)
            
#             data.append({
#                 'timestamp': current_date,
#                 'aqi': aqi_value,
#                 'pm25': pm25,
#                 'pm10': pm10,
#                 'city': 'Lahore',
#                 'country': 'Pakistan'
#             })
            
#             current_date += timedelta(hours=1)
            
#         return pd.DataFrame(data)
    
#     def fetch_current_aqi_data(self):
#         """Fetch current AQI data from AQICN API with multiple station fallback"""
#         # Try different monitoring stations in Lahore
#         stations = [
#             "lahore",
#             "@11765",  # US Embassy station ID
#             "lahore/us-embassy",
#             "lahore,pk",
#             "pakistan/lahore",
#             "lahore/mohlanwal",  # PGSHF Lahore station
#             "lahore/punjab",
#             "@A211000"  # PGSHF Lahore Mohlanwal station ID
#         ]
        
#         for station in stations:
#             url = f"https://api.waqi.info/feed/{station}/?token={self.aqi_api_key}"
            
#             try:
#                 print(f"Trying station: {station}")
#                 response = requests.get(url, timeout=10)
                
#                 if response.status_code == 200:
#                     data = response.json()
                    
#                     if data["status"] == "ok":
#                         aqi = data["data"]["aqi"]
#                         iaqi = data["data"].get("iaqi", {})
                        
#                         # Get PM2.5
#                         pm25 = iaqi.get("pm25", {}).get("v", None)
                        
#                         # Try to get PM10 from current readings first
#                         pm10 = iaqi.get("pm10", {}).get("v", None)
                        
#                         # If PM10 not available in current readings, try forecast data
#                         if pm10 is None and "forecast" in data["data"]:
#                             forecast = data["data"]["forecast"]
#                             if "daily" in forecast and "pm10" in forecast["daily"]:
#                                 pm10_forecast = forecast["daily"]["pm10"]
#                                 if pm10_forecast:
#                                     # Use today's average PM10 from forecast
#                                     today = datetime.now().strftime("%Y-%m-%d")
#                                     for day_data in pm10_forecast:
#                                         if day_data["day"] == today:
#                                             pm10 = day_data.get("avg")
#                                             print(f"Using PM10 from today's forecast: {pm10}")
#                                             break
#                                     # If today not found, use the first available forecast
#                                     if pm10 is None and pm10_forecast:
#                                         pm10 = pm10_forecast[0].get("avg")
#                                         print(f"Using PM10 from forecast (first available): {pm10}")
                        
#                         # Get timestamp from API
#                         api_time = data["data"].get("time", {})
#                         if "iso" in api_time:
#                             timestamp = datetime.fromisoformat(api_time["iso"])
#                         elif "s" in api_time:
#                             timestamp = datetime.fromisoformat(api_time["s"].replace("Z", "+00:00"))
#                         else:
#                             timestamp = datetime.now()
                        
#                         # Check if data is too old (more than 6 hours)
#                         data_age = datetime.now() - timestamp.replace(tzinfo=None)
#                         if data_age.total_seconds() > 6 * 3600:  # 6 hours
#                             print(f"⚠️  Data is {data_age} old from {timestamp}")
#                             print(f"Data too stale, trying next station...")
#                             continue  # Skip this station and try the next one
                        
#                         station_name = data["data"]["city"]["name"]
#                         print(f"✅ Found FRESH data from: {station_name}")
#                         print(f"Data timestamp: {timestamp}")
#                         print(f"AQI: {aqi}, PM2.5: {pm25}, PM10: {pm10}")
                        
#                         df = pd.DataFrame([{
#                             'timestamp': timestamp,
#                             'aqi': aqi,
#                             'pm25': pm25,
#                             'pm10': pm10,
#                             'city': 'Lahore',
#                             'country': 'Pakistan'
#                         }])
                        
#                         return df
                
#             except Exception as e:
#                 print(f"Error with station {station}: {e}")
#                 continue
        
#         print("All AQICN stations failed, trying IQAir API...")
        
#         # Try IQAir API as backup
#         try:
#             # IQAir API endpoint (you'll need to get a free API key from https://www.iqair.com/air-pollution-data-api)
#             iqair_key = "58eadfcf-e27b-497e-a344-7ee23d1e12c1"  # Your IQAir API key
#             iqair_url = f"http://api.airvisual.com/v2/city?city=Lahore&state=Punjab&country=Pakistan&key={iqair_key}"
            
#             print(f"Trying IQAir API: {iqair_url}")
#             response = requests.get(iqair_url, timeout=10)
#             if response.status_code == 200:
#                     data = response.json()
#                     if data["status"] == "success":
#                         current = data["data"]["current"]
#                         pollution = current["pollution"]
                        
#                         aqi = pollution["aqius"]
#                         # Extract PM2.5 and PM10 from pollution data
#                         pm25 = None
#                         pm10 = None
                        
#                         # IQAir API structure might vary, check available keys
#                         print(f"Available pollution data: {list(pollution.keys())}")
                        
#                         # Try to get PM values
#                         if "pm25" in pollution:
#                             pm25 = pollution["pm25"]
#                         elif "pm2p5" in pollution:
#                             pm25 = pollution["pm2p5"]
                            
#                         if "pm10" in pollution:
#                             pm10 = pollution["pm10"]
                        
#                         # Use reasonable fallbacks if not available
#                         if pm25 is None:
#                             pm25 = round(aqi * 0.6)  # Rough conversion estimate
#                         if pm10 is None:
#                             pm10 = round(pm25 * 1.4)  # Rough conversion estimate
                        
#                         timestamp = datetime.now()
                        
#                         print(f"✅ Got fresh data from IQAir API")
#                         print(f"AQI: {aqi}, PM2.5: {pm25}, PM10: {pm10}")
                        
#                         return pd.DataFrame([{
#                             'timestamp': timestamp,
#                             'aqi': aqi,
#                             'pm25': pm25,
#                             'pm10': pm10,
#                             'city': 'Lahore',
#                             'country': 'Pakistan'
#                         }])
#         except Exception as e:
#             print(f"IQAir API also failed: {e}")
        
#         print("All APIs failed, using fallback simulated data")
#         return self._get_fallback_aqi_data()

#     def _get_fallback_aqi_data(self):
#         """Generate fallback AQI data when API fails"""
#         print("Using fallback simulated data")
#         aqi_value = np.random.randint(50, 300)
#         pm25 = np.random.uniform(20, 150)
#         pm10 = np.random.uniform(30, 200)
        
#         return pd.DataFrame([{
#             'timestamp': datetime.now(),
#             'aqi': aqi_value,
#             'pm25': pm25,
#             'pm10': pm10,
#             'city': 'Lahore',
#             'country': 'Pakistan'
#         }])

#     def fetch_weather_data(self, start_date=None, end_date=None):
#         """Fetch weather data with better error handling"""
#         if start_date and end_date:
#             # Simulate Historical Weather Data
#             data = []
#             current_date = start_date
#             while current_date <= end_date:
#                 temp = np.random.uniform(25, 45)
#                 humidity = np.random.uniform(30, 80)
#                 wind_speed = np.random.uniform(2, 15)
#                 pressure = np.random.uniform(1010, 1020)

#                 data.append({
#                     'timestamp': current_date,
#                     'temperature': temp,
#                     'humidity': humidity,
#                     'pressure': pressure,
#                     'wind_speed': wind_speed
#                 })
#                 current_date += timedelta(hours=1)

#             return pd.DataFrame(data)

#         else:
#             # Fetch Current Weather from OpenWeather API
#             lat, lon = self.lahore_coords["lat"], self.lahore_coords["lon"]
#             url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"

#             try:
#                 print(f"Fetching weather data from: {url}")
#                 response = requests.get(url, timeout=10)
#                 print(f"Weather API status code: {response.status_code}")
                
#                 if response.status_code == 200:
#                     data = response.json()
#                     main = data["main"]
#                     wind = data["wind"]

#                     df = pd.DataFrame([{
#                         'timestamp': datetime.now(),
#                         'temperature': main.get('temp'),
#                         'humidity': main.get('humidity'),
#                         'pressure': main.get('pressure'),
#                         'wind_speed': wind.get('speed', 0)
#                     }])
#                     print(f"Successfully fetched weather data")
#                     return df
#                 else:
#                     print(f"Failed to fetch weather data. Status code: {response.status_code}")
#                     print(f"Response: {response.text}")
#                     return self._get_fallback_weather_data()
                    
#             except Exception as e:
#                 print(f"Weather API error: {e}")
#                 return self._get_fallback_weather_data()

#     def _get_fallback_weather_data(self):
#         """Generate fallback weather data when API fails"""
#         print("Using fallback weather data")
#         return pd.DataFrame([{
#             'timestamp': datetime.now(),
#             'temperature': np.random.uniform(25, 45),
#             'humidity': np.random.uniform(30, 80),
#             'pressure': np.random.uniform(1010, 1020),
#             'wind_speed': np.random.uniform(2, 15)
#         }])
    
#     def save_data_to_csv(self, data, filename):
#         """Save data to CSV with better error handling"""
#         if data.empty:
#             print("No data to save")
#             return
            
#         base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         raw_data_dir = os.path.join(base_dir, 'data', 'raw')
#         os.makedirs(raw_data_dir, exist_ok=True)
        
#         filepath = os.path.join(raw_data_dir, filename)

#         try:
#             if os.path.exists(filepath):
#                 # Check if the new data is different from the last entry
#                 existing_data = pd.read_csv(filepath)
#                 if not existing_data.empty:
#                     last_row = existing_data.iloc[-1]
#                     new_row = data.iloc[0]
                    
#                     # Compare key values (excluding timestamp)
#                     if 'aqi' in new_row and 'aqi' in last_row:
#                         if (new_row['aqi'] == last_row['aqi'] and 
#                             new_row.get('pm25') == last_row.get('pm25') and 
#                             new_row.get('pm10') == last_row.get('pm10')):
#                             print("Data unchanged - skipping duplicate entry")
#                             return
                
#                 data.to_csv(filepath, mode='a', header=False, index=False)
#             else:
#                 data.to_csv(filepath, index=False)

#             print(f"Data saved to {filepath}")
#             print(f"Saved data: {data.to_dict('records')[0]}")
            
#         except Exception as e:
#             print(f"Error saving data: {e}")

#     def find_all_lahore_stations(self):
#         """Find all available AQI monitoring stations in Lahore"""
#         search_url = f"https://api.waqi.info/search/?token={self.aqi_api_key}&keyword=lahore"
        
#         try:
#             response = requests.get(search_url, timeout=10)
#             if response.status_code == 200:
#                 data = response.json()
#                 if data["status"] == "ok":
#                     print("🔍 Available Lahore monitoring stations:")
#                     for station in data["data"]:
#                         station_id = station.get("uid")
#                         station_name = station.get("station", {}).get("name", "Unknown")
#                         aqi = station.get("aqi", "N/A")
#                         print(f"  • ID: @{station_id} | Name: {station_name} | AQI: {aqi}")
#                     return data["data"]
#         except Exception as e:
#             print(f"Error searching stations: {e}")
#         return []

#     def test_api_connection(self):
#         """Test API connections and find best monitoring station"""
#         print("Testing API connections...")
        
#         # Find all available stations
#         stations = self.find_all_lahore_stations()
        
#         # Test AQICN API with current default
#         city = "lahore"
#         aqi_url = f"https://api.waqi.info/feed/{city}/?token={self.aqi_api_key}"
        
#         try:
#             response = requests.get(aqi_url, timeout=10)
#             print(f"\n📊 AQICN API - Status: {response.status_code}")
#             if response.status_code == 200:
#                 data = response.json()
#                 print(f"AQICN API - Response status: {data.get('status')}")
#                 if data.get('status') == 'ok':
#                     aqi = data['data']['aqi']
#                     timestamp = data['data'].get('time', {}).get('s', 'Unknown')
#                     station_name = data['data']['city']['name']
                    
#                     print(f"Current Station: {station_name}")
#                     print(f"AQI: {aqi} | Last Update: {timestamp}")
#                     print(f"Available pollutants: {list(data['data']['iaqi'].keys())}")
                    
#                     # Check data freshness
#                     if 'time' in data['data'] and 's' in data['data']['time']:
#                         try:
#                             data_time = datetime.fromisoformat(data['data']['time']['s'].replace('Z', '+00:00'))
#                             age = datetime.now() - data_time.replace(tzinfo=None)
#                             print(f"⏰ Data age: {age}")
#                             if age.total_seconds() > 3600:  # More than 1 hour
#                                 print("⚠️  Data seems stale - you might want to try a different station")
#                         except:
#                             pass
                            
#         except Exception as e:
#             print(f"AQICN API test failed: {e}")
        
#         # Test OpenWeather API
#         lat, lon = self.lahore_coords["lat"], self.lahore_coords["lon"]
#         weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"
        
#         try:
#             response = requests.get(weather_url, timeout=10)
#             print(f"\n🌤️  OpenWeather API - Status: {response.status_code}")
#             if response.status_code == 200:
#                 data = response.json()
#                 print(f"Current Temperature: {data['main']['temp']}°C")
#                 print(f"Humidity: {data['main']['humidity']}% | Wind: {data['wind'].get('speed', 0)} m/s")
#         except Exception as e:
#             print(f"OpenWeather API test failed: {e}")
        
#     def run_backfill(self, start_date_str="2024-07-01"):
#         """Run backfill process from July 1st to today"""
#         start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
#         end_date = datetime.now()

#         print(f"Starting backfill from {start_date} to {end_date}")

#         # Fetch AQI data (simulated)
#         aqi_data = self.fetch_historical_aqi_data(start_date, end_date)
#         self.save_data_to_csv(aqi_data, f"aqi_data_backfill_{start_date.strftime('%Y%m%d')}.csv")

#         # Fetch Weather data (simulated)
#         weather_data = self.fetch_weather_data(start_date, end_date)
#         self.save_data_to_csv(weather_data, f"weather_data_backfill_{start_date.strftime('%Y%m%d')}.csv")

#         print("Backfill completed successfully!")

#     def run_live_fetch(self):
#         """Fetch current data for live mode"""
#         print("Fetching current data...")
        
#         # Fetch current AQI
#         current_aqi = self.fetch_current_aqi_data()
#         if not current_aqi.empty:
#             self.save_data_to_csv(current_aqi, "aqi_data_live.csv")

#         # Fetch current weather
#         current_weather = self.fetch_weather_data()
#         if not current_weather.empty:
#             self.save_data_to_csv(current_weather, "weather_data_live.csv")
        
#         print("Live data fetch completed!")

# if __name__ == "__main__":
#     fetcher = AQIDataFetcher()
    
#     # Test API connections first
#     fetcher.test_api_connection()
    
#     # Run backfill for historical data (commented out to focus on live data)
#     # fetcher.run_backfill()
    
#     # Fetch current data
#     fetcher.run_live_fetch()
    
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

class AQIDataFetcher:
    def __init__(self):
        # Using free APIs - replace with actual keys
        self.aqi_api_key = "03f59b0cc6f536e1645c54fe248c0a7d5d9f04a2"  # Replace with actual AQICN API key
        self.weather_api_key = "e9b863901ee71e7b2e720bf36e69ca87"  # Replace with actual OpenWeather API key
        self.lahore_coords = {"lat": 31.5204, "lon": 74.3587}
        
        # Add validation thresholds
        self.MAX_REASONABLE_AQI = 400  # Anything above this is suspicious
        self.MAX_REASONABLE_PM25 = 300  # µg/m³
        self.MAX_REASONABLE_PM10 = 500  # µg/m³
        
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
        
    def validate_aqi_reading(self, aqi, pm25, pm10, station_name):
        """Validate AQI readings for obvious errors"""
        issues = []
        
        # Check for unreasonable values
        if aqi and aqi > self.MAX_REASONABLE_AQI:
            issues.append(f"AQI {aqi} exceeds reasonable threshold ({self.MAX_REASONABLE_AQI})")
            
        if pm25 and pm25 > self.MAX_REASONABLE_PM25:
            issues.append(f"PM2.5 {pm25} exceeds reasonable threshold ({self.MAX_REASONABLE_PM25})")
            
        if pm10 and pm10 > self.MAX_REASONABLE_PM10:
            issues.append(f"PM10 {pm10} exceeds reasonable threshold ({self.MAX_REASONABLE_PM10})")
        
        # Check for impossible PM relationships
        if pm25 and pm10 and pm25 > pm10:
            issues.append(f"PM2.5 ({pm25}) cannot be higher than PM10 ({pm10})")
            
        # Check for extreme inconsistencies
        if aqi and pm25:
            # Rough AQI to PM2.5 conversion check
            expected_pm25_range = self.aqi_to_pm25_range(aqi)
            if pm25 < expected_pm25_range[0] * 0.3 or pm25 > expected_pm25_range[1] * 3:
                issues.append(f"PM2.5 ({pm25}) inconsistent with AQI ({aqi})")
        
        if issues:
            print(f"⚠️  VALIDATION ISSUES for {station_name}:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        return True
    
    def aqi_to_pm25_range(self, aqi):
        """Convert AQI to expected PM2.5 range (rough estimate)"""
        if aqi <= 50:
            return (0, 12)
        elif aqi <= 100:
            return (12, 35)
        elif aqi <= 150:
            return (35, 55)
        elif aqi <= 200:
            return (55, 150)
        elif aqi <= 300:
            return (150, 250)
        else:
            return (250, 500)
    
    def fetch_current_aqi_data(self):
        """Fetch current AQI data with enhanced validation and cross-checking"""
        # Prioritize reliable stations first
        stations = [
            "lahore/us-embassy",  # Usually most reliable
            "lahore",             # General city reading
            "@11765",            # US Embassy station ID
            "lahore,pk",
            "pakistan/lahore",
            "lahore/punjab",
            "lahore/mohlanwal",  # PGSHF stations - less reliable
            "@A211000"          # PGSHF Lahore Mohlanwal station ID
        ]
        
        valid_readings = []
        
        for station in stations:
            url = f"https://api.waqi.info/feed/{station}/?token={self.aqi_api_key}"
            
            try:
                print(f"Trying station: {station}")
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data["status"] == "ok":
                        aqi = data["data"]["aqi"]
                        iaqi = data["data"].get("iaqi", {})
                        
                        # Get PM2.5
                        pm25 = iaqi.get("pm25", {}).get("v", None)
                        
                        # Try to get PM10 from current readings first
                        pm10 = iaqi.get("pm10", {}).get("v", None)
                        
                        # If PM10 not available in current readings, try forecast data
                        if pm10 is None and "forecast" in data["data"]:
                            forecast = data["data"]["forecast"]
                            if "daily" in forecast and "pm10" in forecast["daily"]:
                                pm10_forecast = forecast["daily"]["pm10"]
                                if pm10_forecast:
                                    today = datetime.now().strftime("%Y-%m-%d")
                                    for day_data in pm10_forecast:
                                        if day_data["day"] == today:
                                            pm10 = day_data.get("avg")
                                            print(f"Using PM10 from today's forecast: {pm10}")
                                            break
                                    if pm10 is None and pm10_forecast:
                                        pm10 = pm10_forecast[0].get("avg")
                                        print(f"Using PM10 from forecast (first available): {pm10}")
                        
                        # Get timestamp
                        api_time = data["data"].get("time", {})
                        if "iso" in api_time:
                            timestamp = datetime.fromisoformat(api_time["iso"])
                        elif "s" in api_time:
                            timestamp = datetime.fromisoformat(api_time["s"].replace("Z", "+00:00"))
                        else:
                            timestamp = datetime.now()
                        
                        # Check if data is too old
                        data_age = datetime.now() - timestamp.replace(tzinfo=None)
                        if data_age.total_seconds() > 6 * 3600:
                            print(f"⚠️  Data is {data_age} old from {timestamp}")
                            print(f"Data too stale, trying next station...")
                            continue
                        
                        station_name = data["data"]["city"]["name"]
                        
                        # VALIDATE THE READING
                        if self.validate_aqi_reading(aqi, pm25, pm10, station_name):
                            print(f"✅ Found VALID data from: {station_name}")
                            print(f"Data timestamp: {timestamp}")
                            print(f"AQI: {aqi}, PM2.5: {pm25}, PM10: {pm10}")
                            
                            valid_readings.append({
                                'station': station_name,
                                'station_id': station,
                                'timestamp': timestamp,
                                'aqi': aqi,
                                'pm25': pm25,
                                'pm10': pm10,
                                'data_age_hours': data_age.total_seconds() / 3600
                            })
                            
                            # If we found a reliable station (US Embassy), use it immediately
                            if 'embassy' in station.lower():
                                break
                        else:
                            print(f"❌ INVALID reading from {station_name} - trying next station...")
                            continue
                
            except Exception as e:
                print(f"Error with station {station}: {e}")
                continue
        
        # If no valid AQICN readings found, try IQAir API
        if not valid_readings:
            print("No valid readings from AQICN stations, trying IQAir API...")
            iqair_result = self._try_iqair_api()
            if not iqair_result.empty:
                return iqair_result
            
            print("All APIs failed, using fallback simulated data")
            return self._get_fallback_aqi_data()
        
        # Choose the best reading from valid AQICN readings
        # Sort by reliability (embassy first) and data freshness
        def station_priority(reading):
            if 'embassy' in reading['station'].lower():
                return (0, reading['data_age_hours'])  # Highest priority
            elif 'pgshf' in reading['station'].lower() or 'mohlanwal' in reading['station'].lower():
                return (2, reading['data_age_hours'])  # Lower priority
            else:
                return (1, reading['data_age_hours'])  # Medium priority
        
        valid_readings.sort(key=station_priority)
        best_reading = valid_readings[0]
        
        print(f"\nSELECTED READING: {best_reading['station']}")
        print(f"AQI: {best_reading['aqi']}, PM2.5: {best_reading['pm25']}, PM10: {best_reading['pm10']}")
        
        # Show comparison if multiple readings available
        if len(valid_readings) > 1:
            print(f"\nCOMPARISON OF VALID READINGS:")
            for reading in valid_readings:
                print(f"  {reading['station']}: AQI={reading['aqi']}, PM2.5={reading['pm25']}")
        
        # Also try IQAir for comparison even if we have AQICN data
        print(f"\nAlso checking IQAir for comparison...")
        iqair_result = self._try_iqair_api_for_comparison()
        
        df = pd.DataFrame([{
            'timestamp': best_reading['timestamp'],
            'aqi': best_reading['aqi'],
            'pm25': best_reading['pm25'],
            'pm10': best_reading['pm10'],
            'city': 'Lahore',
            'country': 'Pakistan'
        }])
        
        return df
    
    def _try_iqair_api_for_comparison(self):
        """Try IQAir API for comparison with AQICN readings"""
        try:
            iqair_key = "58eadfcf-e27b-497e-a344-7ee23d1e12c1"
            iqair_url = f"http://api.airvisual.com/v2/city?city=Lahore&state=Punjab&country=Pakistan&key={iqair_key}"
            
            response = requests.get(iqair_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    current = data["data"]["current"]
                    pollution = current["pollution"]
                    
                    aqi = pollution["aqius"]
                    pm25 = pollution.get("pm25")
                    pm10 = pollution.get("pm10")
                    
                    if self.validate_aqi_reading(aqi, pm25, pm10, "IQAir"):
                        print(f"IQAir comparison: AQI={aqi}, PM2.5={pm25}, PM10={pm10}")
                    else:
                        print("IQAir reading failed validation for comparison")
                        
        except Exception as e:
            print(f"IQAir comparison check failed: {e}")
    
    def _try_iqair_api(self):
        """Try IQAir API as backup with validation"""
        try:
            iqair_key = "58eadfcf-e27b-497e-a344-7ee23d1e12c1"
            iqair_url = f"http://api.airvisual.com/v2/city?city=Lahore&state=Punjab&country=Pakistan&key={iqair_key}"
            
            print(f"Trying IQAir API...")
            response = requests.get(iqair_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    current = data["data"]["current"]
                    pollution = current["pollution"]
                    
                    aqi = pollution["aqius"]
                    pm25 = pollution.get("pm25")
                    pm10 = pollution.get("pm10")
                    
                    if self.validate_aqi_reading(aqi, pm25, pm10, "IQAir"):
                        print(f"Got valid data from IQAir API")
                        print(f"AQI: {aqi}, PM2.5: {pm25}, PM10: {pm10}")
                        
                        return pd.DataFrame([{
                            'timestamp': datetime.now(),
                            'aqi': aqi,
                            'pm25': pm25 or round(aqi * 0.6),
                            'pm10': pm10 or round((pm25 or aqi * 0.6) * 1.4),
                            'city': 'Lahore',
                            'country': 'Pakistan'
                        }])
                    else:
                        print("IQAir reading failed validation")
        except Exception as e:
            print(f"IQAir API failed: {e}")
        
        return pd.DataFrame()  # Return empty DataFrame if failed

    def _get_fallback_aqi_data(self):
        """Generate fallback AQI data when API fails"""
        print("Using fallback simulated data")
        # Use reasonable Lahore averages
        aqi_value = np.random.randint(80, 150)  # More realistic range
        pm25 = np.random.uniform(30, 80)
        pm10 = np.random.uniform(45, 120)
        
        return pd.DataFrame([{
            'timestamp': datetime.now(),
            'aqi': aqi_value,
            'pm25': pm25,
            'pm10': pm10,
            'city': 'Lahore',
            'country': 'Pakistan'
        }])

    def fetch_weather_data(self, start_date=None, end_date=None):
        """Fetch weather data with better error handling"""
        if start_date and end_date:
            # Historical weather simulation
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
            # Current weather
            lat, lon = self.lahore_coords["lat"], self.lahore_coords["lon"]
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"

            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    main = data["main"]
                    wind = data["wind"]

                    return pd.DataFrame([{
                        'timestamp': datetime.now(),
                        'temperature': main.get('temp'),
                        'humidity': main.get('humidity'),
                        'pressure': main.get('pressure'),
                        'wind_speed': wind.get('speed', 0)
                    }])
                else:
                    return self._get_fallback_weather_data()
            except Exception as e:
                print(f"Weather API error: {e}")
                return self._get_fallback_weather_data()

    def _get_fallback_weather_data(self):
        """Generate fallback weather data when API fails"""
        return pd.DataFrame([{
            'timestamp': datetime.now(),
            'temperature': np.random.uniform(25, 45),
            'humidity': np.random.uniform(30, 80),
            'pressure': np.random.uniform(1010, 1020),
            'wind_speed': np.random.uniform(2, 15)
        }])
    
    def save_data_to_csv(self, data, filename):
        """Save data to CSV with duplicate checking"""
        if data.empty:
            print("No data to save")
            return
            
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_data_dir = os.path.join(base_dir, 'data', 'raw')
        os.makedirs(raw_data_dir, exist_ok=True)
        
        filepath = os.path.join(raw_data_dir, filename)

        try:
            if os.path.exists(filepath):
                existing_data = pd.read_csv(filepath)
                if not existing_data.empty:
                    last_row = existing_data.iloc[-1]
                    new_row = data.iloc[0]
                    
                    # Compare key values (excluding timestamp)
                    if ('aqi' in new_row and 'aqi' in last_row and
                        new_row['aqi'] == last_row['aqi'] and 
                        new_row.get('pm25') == last_row.get('pm25') and 
                        new_row.get('pm10') == last_row.get('pm10')):
                        print("Data unchanged - skipping duplicate entry")
                        return
                
                data.to_csv(filepath, mode='a', header=False, index=False)
            else:
                data.to_csv(filepath, index=False)

            print(f"Data saved to {filepath}")
            print(f"Saved data: {data.to_dict('records')[0]}")
            
        except Exception as e:
            print(f"Error saving data: {e}")

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

    def find_all_lahore_stations(self):
        """Find all available AQI monitoring stations in Lahore"""
        search_url = f"https://api.waqi.info/search/?token={self.aqi_api_key}&keyword=lahore"
        
        try:
            response = requests.get(search_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "ok":
                    print("Available Lahore monitoring stations:")
                    for station in data["data"]:
                        station_id = station.get("uid")
                        station_name = station.get("station", {}).get("name", "Unknown")
                        aqi = station.get("aqi", "N/A")
                        print(f"  • ID: @{station_id} | Name: {station_name} | AQI: {aqi}")
                    return data["data"]
        except Exception as e:
            print(f"Error searching stations: {e}")
        return []

    def test_api_connection(self):
        """Test API connections and find best monitoring station"""
        print("Testing API connections...")
        
        # Find all available stations
        stations = self.find_all_lahore_stations()
        
        # Test AQICN API with current default
        city = "lahore"
        aqi_url = f"https://api.waqi.info/feed/{city}/?token={self.aqi_api_key}"
        
        try:
            response = requests.get(aqi_url, timeout=10)
            print(f"\nAQICN API - Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"AQICN API - Response status: {data.get('status')}")
                if data.get('status') == 'ok':
                    aqi = data['data']['aqi']
                    timestamp = data['data'].get('time', {}).get('s', 'Unknown')
                    station_name = data['data']['city']['name']
                    
                    print(f"Current Station: {station_name}")
                    print(f"AQI: {aqi} | Last Update: {timestamp}")
                    print(f"Available pollutants: {list(data['data']['iaqi'].keys())}")
                    
                    # Check data freshness
                    if 'time' in data['data'] and 's' in data['data']['time']:
                        try:
                            data_time = datetime.fromisoformat(data['data']['time']['s'].replace('Z', '+00:00'))
                            age = datetime.now() - data_time.replace(tzinfo=None)
                            print(f"Data age: {age}")
                            if age.total_seconds() > 3600:  # More than 1 hour
                                print("Data seems stale - you might want to try a different station")
                        except:
                            pass
                            
        except Exception as e:
            print(f"AQICN API test failed: {e}")
        
        # Test OpenWeather API
        lat, lon = self.lahore_coords["lat"], self.lahore_coords["lon"]
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"
        
        try:
            response = requests.get(weather_url, timeout=10)
            print(f"\nOpenWeather API - Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Current Temperature: {data['main']['temp']}°C")
                print(f"Humidity: {data['main']['humidity']}% | Wind: {data['wind'].get('speed', 0)} m/s")
        except Exception as e:
            print(f"OpenWeather API test failed: {e}")

    def run_live_fetch(self):
        """Fetch current data for live mode with validation"""
        print("=== Fetching current data with validation ===")
        
        # Fetch current AQI with validation
        current_aqi = self.fetch_current_aqi_data()
        if not current_aqi.empty:
            self.save_data_to_csv(current_aqi, "aqi_data_live.csv")
            
            # Show summary
            row = current_aqi.iloc[0]
            print(f"\nFINAL SELECTED DATA:")
            print(f"   Source: {row.get('source_station', 'Unknown')}")
            print(f"   AQI: {row['aqi']}")
            print(f"   PM2.5: {row['pm25']} µg/m³")
            print(f"   PM10: {row['pm10']} µg/m³")
            print(f"   Timestamp: {row['timestamp']}")

        # Fetch current weather
        current_weather = self.fetch_weather_data()
        if not current_weather.empty:
            self.save_data_to_csv(current_weather, "weather_data_live.csv")
        
        print("Live data fetch completed!")

if __name__ == "__main__":
    fetcher = AQIDataFetcher()
    
    # Test API connections first
    fetcher.test_api_connection()
    
    # Run backfill for historical data (commented out to focus on live data)
    # fetcher.run_backfill()
    
    # Fetch current data
    fetcher.run_live_fetch()
    