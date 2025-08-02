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
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
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
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['heat_index'] = df['temperature'] + 0.5 * df['humidity']
        if 'pm25' in df.columns and 'pm10' in df.columns:
            df['pm_ratio'] = df['pm25'] / (df['pm10'] + 1e-6)
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
        aqi_data['timestamp'] = pd.to_datetime(aqi_data['timestamp'], format='mixed')
        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'], format='mixed')
        aqi_data['timestamp_rounded'] = aqi_data['timestamp'].dt.round('H')
        weather_data['timestamp_rounded'] = weather_data['timestamp'].dt.round('H')
        merged_data = pd.merge(aqi_data, weather_data, left_on='timestamp_rounded', right_on='timestamp_rounded',
                               how='inner', suffixes=('', '_weather'))
        merged_data['timestamp'] = merged_data['timestamp']
        merged_data = merged_data.drop(['timestamp_rounded', 'timestamp_weather'], axis=1)
        return merged_data
    
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

        print("One-Hot Encoding aqi_category...")
        merged_data['aqi_category'] = merged_data['aqi_category'].astype(str)
        merged_data = pd.get_dummies(merged_data, columns=['aqi_category'], prefix='aqi_category')
        expected_categories = [f'aqi_category_{i}' for i in range(6)]
        for cat in expected_categories:
            if cat not in merged_data.columns:
                merged_data[cat] = 0

        print("Creating target variables...")
        merged_data = self.create_target_variables(merged_data)
        
        merged_data = merged_data.dropna(subset=['aqi'])
        
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
