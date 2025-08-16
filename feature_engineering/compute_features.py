#feature_engineering(compute_feature.py)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob

class FeatureEngineer:
    def __init__(self):
        # Dynamically set paths relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        self.raw_data_path = os.path.join(project_root, "data/raw/")
        self.features_path = os.path.join(project_root, "data/features/")
        
    def load_raw_data(self):
        """Load all raw data files and combine them"""
        aqi_files = glob.glob(os.path.join(self.raw_data_path, "aqi_data_*.csv"))
        weather_files = glob.glob(os.path.join(self.raw_data_path, "weather_data_*.csv"))
        
        aqi_data_list = [pd.read_csv(file) for file in aqi_files]
        aqi_data = pd.concat(aqi_data_list, ignore_index=True) if aqi_data_list else pd.DataFrame()
        
        weather_data_list = [pd.read_csv(file) for file in weather_files]
        weather_data = pd.concat(weather_data_list, ignore_index=True) if weather_data_list else pd.DataFrame()
        
        return aqi_data, weather_data
    
    def create_time_features(self, df):
        # FIXED: Added utc=True to handle mixed timezones
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        
        # Convert to local timezone if needed (optional)
        # df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Karachi')
        
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        # Add season feature
        df['season'] = ((df['month'] % 12 + 3) // 3).map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
        return df
    
    def create_lag_features(self, df, columns, lags=[1, 6, 12, 24]):
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        for col in columns:
            for lag in lags:
                df_sorted[f'{col}_lag_{lag}'] = df_sorted[col].shift(lag)
        return df_sorted
    
    def create_rolling_features(self, df, columns, windows=[6, 12, 24]):
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        for col in columns:
            for window in windows:
                df_sorted[f'{col}_rolling_mean_{window}'] = df_sorted[col].rolling(window=window).mean()
                df_sorted[f'{col}_rolling_std_{window}'] = df_sorted[col].rolling(window=window).std()
                df_sorted[f'{col}_rolling_max_{window}'] = df_sorted[col].rolling(window=window).max()
                df_sorted[f'{col}_rolling_min_{window}'] = df_sorted[col].rolling(window=window).min()
        return df_sorted
    
    def create_derived_features(self, df):
        if 'aqi' in df.columns:
            df['aqi_change_rate'] = df['aqi'].pct_change()
            df['aqi_category'] = pd.cut(df['aqi'], bins=[0, 50, 100, 150, 200, 300, float('inf')],
                                        labels=[0, 1, 2, 3, 4, 5])
        
        # Better heat index calculation
        if 'temperature' in df.columns and 'humidity' in df.columns:
            # More accurate heat index approximation
            T = df['temperature']
            H = df['humidity']
            df['heat_index'] = T + (0.5 * (H - 10)) / 10
            df['comfort_index'] = T - 0.4 * (T - 10) * (1 - H/100)
        
        if 'pm25' in df.columns and 'pm10' in df.columns:
            df['pm_ratio'] = df['pm25'] / (df['pm10'] + 1e-6)
            
        # Wind chill factor
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            df['wind_chill'] = df['temperature'] - (df['wind_speed'] * 0.7)
            
        return df
    
    def create_target_variables(self, df):
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        if 'aqi' in df.columns:
            df_sorted['target_aqi_24h'] = df_sorted['aqi'].shift(-24)
            df_sorted['target_aqi_48h'] = df_sorted['aqi'].shift(-48)
            df_sorted['target_aqi_72h'] = df_sorted['aqi'].shift(-72)
            df_sorted['target_aqi_3day_avg'] = df_sorted[['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']].mean(axis=1)
        return df_sorted
    
    def merge_aqi_weather_data(self, aqi_data, weather_data):
        # FIXED: Added utc=True to handle mixed timezones
        aqi_data['timestamp'] = pd.to_datetime(aqi_data['timestamp'], format='mixed', utc=True)
        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'], format='mixed', utc=True)
        
        aqi_data['timestamp_rounded'] = aqi_data['timestamp'].dt.round('H')
        weather_data['timestamp_rounded'] = weather_data['timestamp'].dt.round('H')
        merged_data = pd.merge(aqi_data, weather_data, left_on='timestamp_rounded', right_on='timestamp_rounded',
                               how='inner', suffixes=('', '_weather'))
        merged_data['timestamp'] = merged_data['timestamp']
        merged_data = merged_data.drop(['timestamp_rounded', 'timestamp_weather'], axis=1)
        
        # Remove duplicates after merging
        merged_data = merged_data.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        return merged_data
    
    def handle_nulls_and_outliers(self, df):
        """Handle null values and outliers"""
        print("Handling null values and outliers...")
        
        # Remove columns that are completely null
        df = df.dropna(axis=1, how='all')
        
        # Remove columns with more than 80% nulls
        null_threshold = len(df) * 0.8
        df = df.dropna(axis=1, thresh=null_threshold)
        
        # Handle outliers in key columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['aqi', 'pm25', 'pm10']:  # AQI related columns
                # Remove extreme outliers (beyond 99.9 percentile)
                upper_limit = df[col].quantile(0.999)
                df = df[df[col] <= upper_limit]
            elif col in ['temperature', 'humidity', 'pressure', 'wind_speed']:  # Weather columns
                # Remove values outside reasonable ranges
                if col == 'temperature':
                    df = df[(df[col] >= -50) & (df[col] <= 60)]  # Reasonable temp range
                elif col == 'humidity':
                    df = df[(df[col] >= 0) & (df[col] <= 100)]  # Valid humidity range
                elif col == 'pressure':
                    df = df[(df[col] >= 900) & (df[col] <= 1100)]  # Valid pressure range
                elif col == 'wind_speed':
                    df = df[(df[col] >= 0) & (df[col] <= 200)]  # Valid wind speed range
        
        print(f"Data shape after outlier removal: {df.shape}")
        return df
    
    def compute_all_features(self):
        print("Loading raw data...")
        aqi_data, weather_data = self.load_raw_data()
        
        if aqi_data.empty or weather_data.empty:
            print("No raw data found. Please run data fetching first.")
            return None
            
        print("Merging AQI and weather data...")
        merged_data = self.merge_aqi_weather_data(aqi_data, weather_data)
        
        print("Creating time-based features...")
        merged_data = self.create_time_features(merged_data)
        
        print("Creating lag features...")
        numeric_columns = ['aqi', 'pm25', 'pm10', 'temperature', 'humidity', 'wind_speed', 'pressure']
        available_columns = [col for col in numeric_columns if col in merged_data.columns]
        merged_data = self.create_lag_features(merged_data, available_columns)
        
        print("Creating rolling features...")
        merged_data = self.create_rolling_features(merged_data, available_columns)
        
        print("Creating derived features...")
        merged_data = self.create_derived_features(merged_data)

        print("One-Hot Encoding categorical features...")
        # Handle aqi_category
        merged_data['aqi_category'] = merged_data['aqi_category'].astype(str)
        merged_data = pd.get_dummies(merged_data, columns=['aqi_category'], prefix='aqi_category')
        expected_categories = [f'aqi_category_{i}' for i in range(6)]
        for cat in expected_categories:
            if cat not in merged_data.columns:
                merged_data[cat] = 0
        
        # Handle season
        if 'season' in merged_data.columns:
            merged_data = pd.get_dummies(merged_data, columns=['season'], prefix='season')

        print("Creating target variables...")
        merged_data = self.create_target_variables(merged_data)
        
        # Handle nulls and outliers
        merged_data = self.handle_nulls_and_outliers(merged_data)
        
        # Drop rows where essential columns are null
        essential_columns = ['aqi']  # Add more if needed
        merged_data = merged_data.dropna(subset=essential_columns)
        
        # FIXED: Replace deprecated fillna method
        # Old: merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
        # New:
        merged_data = merged_data.ffill().bfill()
        
        # Final check - drop any remaining rows with nulls
        merged_data = merged_data.dropna()
        
        print(f"Final data shape after all processing: {merged_data.shape}")
        print(f"Null values remaining: {merged_data.isnull().sum().sum()}")
        
        if not os.path.exists(self.features_path):
            os.makedirs(self.features_path, exist_ok=True)
        
        output_file = os.path.join(self.features_path, f"features_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        merged_data.to_csv(output_file, index=False)
        print(f"Features saved to {output_file}")
        
        return merged_data

if __name__ == "__main__":
    engineer = FeatureEngineer()
    features = engineer.compute_all_features()
    
    if features is not None:
        print(f"Feature engineering completed. Shape: {features.shape}")
        print(f"Columns: {list(features.columns)}")
        print(f"Memory usage: {features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")