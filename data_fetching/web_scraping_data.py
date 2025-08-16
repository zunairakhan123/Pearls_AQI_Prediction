#web_scraping_weather_data.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from urllib3.exceptions import InsecureRequestWarning
import warnings

# Suppress SSL warnings for testing
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

class WeatherScraper:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_data_dir = os.path.join(self.base_dir, 'data', 'raw')
        
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        # Enhanced user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        ]
        
        self.session = requests.Session()
        self._setup_session()
        
        # Track attempts
        self.scraping_stats = {
            'success': 0, 
            'failed': 0, 
            'total_records': 0
        }
        
    def _setup_session(self):
        """Setup session with enhanced headers"""
        import random
        
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'DNT': '1',
            'Cache-Control': 'max-age=0'
        })
        
        self.session.verify = False
        
        # Improved retry strategy
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def test_weather_sources(self):
        """Test weather data sources"""
        print("🔍 Testing Weather Data Sources...")
        print("="*50)
        
        # Test Open-Meteo (completely free)
        print("Testing Open-Meteo (Free API)...")
        success = self._test_openmeteo()
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
        
        print("="*50)
        return success
    
    def _test_openmeteo(self):
        """Test Open-Meteo (completely free, no API key needed)"""
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': 31.5204,
                'longitude': 74.3587,
                'current': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'current' in data:
                    print(f"   Current temp: {data['current'].get('temperature_2m', 'N/A')}°C")
                    return True
            return False
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def collect_weather_data_chunked(self, start_year=2020, end_year=2025):
        """Collect weather data efficiently using chunked requests"""
        print(f"🌤️ Collecting weather data from {start_year} to {end_year}...")
        
        all_weather_data = []
        failed_chunks = []
        
        # Process in monthly chunks for better efficiency
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Skip future months
                if year == datetime.now().year and month > datetime.now().month:
                    continue
                if year > datetime.now().year:
                    continue
                
                success = False
                retry_count = 0
                max_retries = 2
                
                while not success and retry_count < max_retries:
                    try:
                        # Get month date range
                        start_date = datetime(year, month, 1)
                        if month == 12:
                            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
                        else:
                            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
                        
                        url = "https://archive-api.open-meteo.com/v1/archive"
                        params = {
                            'latitude': 31.5204,
                            'longitude': 74.3587,
                            'start_date': start_date.strftime('%Y-%m-%d'),
                            'end_date': end_date.strftime('%Y-%m-%d'),
                            'hourly': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m',
                            'timezone': 'Asia/Karachi'
                        }
                        
                        response = self.session.get(url, params=params, timeout=20)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            if 'hourly' in data and data['hourly']['time']:
                                month_data = self._process_weather_chunk(data['hourly'])
                                all_weather_data.extend(month_data)
                                print(f"✅ {year}-{month:02d}: {len(month_data)} records")
                                self.scraping_stats['success'] += 1
                                self.scraping_stats['total_records'] += len(month_data)
                                success = True
                            else:
                                raise Exception("No hourly data in response")
                        else:
                            raise Exception(f"HTTP {response.status_code}")
                    
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"⏳ Retrying {year}-{month:02d} (attempt {retry_count + 1})")
                            time.sleep(3)
                        else:
                            print(f"❌ {year}-{month:02d}: {str(e)}")
                            failed_chunks.append(f"{year}-{month:02d}")
                            self.scraping_stats['failed'] += 1
                
                # Small delay between months
                time.sleep(1)
        
        if failed_chunks:
            print(f"\n⚠️ Failed to collect weather data for: {', '.join(failed_chunks)}")
        
        return pd.DataFrame(all_weather_data)
    
    def _process_weather_chunk(self, hourly_data):
        """Process weather data chunk efficiently"""
        times = hourly_data['time']
        temps = hourly_data.get('temperature_2m', [])
        humidity = hourly_data.get('relative_humidity_2m', [])
        pressure = hourly_data.get('pressure_msl', [])
        wind = hourly_data.get('wind_speed_10m', [])
        
        chunk_data = []
        for i, time_str in enumerate(times):
            timestamp = datetime.fromisoformat(time_str.replace('T', ' '))
            
            chunk_data.append({
                'timestamp': timestamp,
                'temperature': temps[i] if i < len(temps) and temps[i] is not None else np.nan,
                'humidity': humidity[i] if i < len(humidity) and humidity[i] is not None else np.nan,
                'pressure': pressure[i] if i < len(pressure) and pressure[i] is not None else np.nan,
                'wind_speed': wind[i] if i < len(wind) and wind[i] is not None else np.nan
            })
        
        return chunk_data
    
    def save_weather_data(self, weather_df):
        """Save weather data"""
        if weather_df.empty:
            print("❌ No weather data to save")
            return False
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"weather_data_scraped_{timestamp}.csv"
            # Create raw_data_dir inside data
            self.raw_data_dir = os.path.join("data", "raw")
            os.makedirs(self.raw_data_dir, exist_ok=True)

            filepath = os.path.join(self.raw_data_dir, filename)
            weather_df.to_csv(filepath, index=False)
            
            print(f"💾 Weather data saved to: {filepath}")
            print(f"📊 Records: {len(weather_df):,}")
            print(f"📅 Date range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")
            
            if 'temperature' in weather_df.columns:
                temp_stats = weather_df['temperature'].describe()
                print(f"🌡️ Temperature: {temp_stats['min']:.1f}°C to {temp_stats['max']:.1f}°C")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving weather data: {e}")
            return False
    
    def run_weather_collection(self, start_year=2020, end_year=2025):
        """Run weather data collection"""
        print(f"🚀 Starting weather data collection from {start_year} to {end_year}")
        print("=" * 70)
        
        # Collect weather data
        print("\n🌤️ WEATHER DATA COLLECTION")
        weather_df = self.collect_weather_data_chunked(start_year, end_year)
        
        success = False
        if not weather_df.empty:
            success = self.save_weather_data(weather_df)
        
        # Print final statistics
        self.print_collection_stats(success)
        
        return weather_df
    
    def print_collection_stats(self, success):
        """Print collection statistics"""
        print("\n" + "="*70)
        print("📊 COLLECTION STATISTICS")
        print("="*70)
        
        total_attempts = self.scraping_stats['success'] + self.scraping_stats['failed']
        
        if total_attempts > 0:
            success_rate = (self.scraping_stats['success'] / total_attempts) * 100
            print(f"🌤️ Weather Success Rate: {success_rate:.1f}% ({self.scraping_stats['success']}/{total_attempts})")
            print(f"📊 Weather Records: {self.scraping_stats['total_records']:,}")
        
        if success:
            print("🎉 Collection completed successfully!")
            print("📁 Files saved in: data/raw/")
            print("✨ Ready for further processing!")
        else:
            print("❌ Collection failed. Please check connection and try again.")

if __name__ == "__main__":
    scraper = WeatherScraper()
    
    # Test weather sources first
    success = scraper.test_weather_sources()
    
    print("\n" + "="*70)
    print("🔧 WEATHER COLLECTION FEATURES:")
    print("✅ Efficient chunked requests for better reliability")
    print("✅ Improved error handling and retry logic")
    print("✅ Better memory management")
    print("✅ Comprehensive weather metrics (temp, humidity, pressure, wind)")
    
    if success:
        response = input("\n🤔 Start weather data collection? (y/n): ").lower().strip()
        
        if response == 'y':
            weather_df = scraper.run_weather_collection(
                start_year=2020, 
                end_year=2025
            )
        else:
            print("👍 Collection cancelled.")
    else:
        print("❌ No working weather data sources found. Please check internet connection.")