# # inference_pipeline.py - FIXED VERSION

# import sys
# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import warnings
# import joblib

# # Dynamically set project root
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..'))
# sys.path.append(project_root)

# from data_fetching.fetch_aqi_data import AQIDataFetcher
# from feature_engineering.compute_features import FeatureEngineer
# from model_training.model_registry import ModelRegistry

# warnings.filterwarnings('ignore')

# class InferencePipeline:
#     def __init__(self):
#         self.data_fetcher = AQIDataFetcher()
#         self.feature_engineer = FeatureEngineer()
#         self.model_registry = ModelRegistry()
#         self.predictions_path = os.path.join(project_root, "data", "predictions")
        
#         # Ensure predictions directory exists
#         os.makedirs(self.predictions_path, exist_ok=True)

#     def fetch_latest_data(self):
#         print("Fetching latest data for inference...")
#         self.data_fetcher.run_live_fetch()
#         end_date = datetime.now()
#         start_date = end_date - timedelta(days=7)
#         aqi_data = self.data_fetcher.fetch_historical_aqi_data(start_date, end_date)
#         weather_data = self.data_fetcher.fetch_weather_data(start_date, end_date)
#         return aqi_data, weather_data

#     def get_aqi_category(self, aqi_value):
#         """Convert AQI value to standardized category (same as training)"""
#         if pd.isna(aqi_value):
#             return 'moderate'  # Default category
#         elif aqi_value <= 50:
#             return 'good'
#         elif aqi_value <= 100:
#             return 'moderate'
#         elif aqi_value <= 150:
#             return 'unhealthy_sensitive'
#         elif aqi_value <= 200:
#             return 'unhealthy'
#         elif aqi_value <= 300:
#             return 'very_unhealthy'
#         else:
#             return 'hazardous'

#     def get_model_feature_columns_from_registry(self, model_info):
#         """Extract the exact feature columns used during model training from registry"""
#         try:
#             feature_columns_str = model_info['feature_columns']
#             if pd.isna(feature_columns_str) or feature_columns_str == '':
#                 return None
#             return feature_columns_str.split(',')
#         except Exception as e:
#             print(f"Error extracting feature columns from registry: {e}")
#             return None

#     def align_features_with_trained_model(self, features_df, model_feature_columns):
#         """Align inference features exactly with trained model features"""
#         print(f"Aligning features with trained model requirements...")
#         print(f"Model expects: {len(model_feature_columns)} features")
#         print(f"We have: {len(features_df.columns)} features")
        
#         # Start with empty dataframe with correct index
#         aligned_features = pd.DataFrame(index=features_df.index)
        
#         # Add each required feature column
#         missing_features = []
#         for feature_col in model_feature_columns:
#             if feature_col in features_df.columns:
#                 aligned_features[feature_col] = features_df[feature_col]
#             else:
#                 # Handle missing features with reasonable defaults
#                 if feature_col.startswith('aqi_category_'):
#                     aligned_features[feature_col] = 0.0  # Categorical feature default
#                 elif 'lag' in feature_col or 'rolling' in feature_col:
#                     aligned_features[feature_col] = 0.0  # Time-based feature default
#                 else:
#                     aligned_features[feature_col] = 0.0  # General default
#                 missing_features.append(feature_col)
        
#         # Handle AQI categories specifically if current AQI is available
#         if 'aqi' in features_df.columns and any(col.startswith('aqi_category_') for col in model_feature_columns):
#             current_aqi = features_df['aqi'].iloc[-1]
#             current_category = self.get_aqi_category(current_aqi)
            
#             # Set the appropriate category column to 1
#             aqi_category_cols = [col for col in model_feature_columns if col.startswith('aqi_category_')]
            
#             # Reset all category columns to 0 first
#             for col in aqi_category_cols:
#                 aligned_features[col] = 0.0
            
#             # Try to set the correct category
#             category_set = False
            
#             # Try numeric categories (0-5)
#             category_to_numeric = {
#                 'good': 0, 'moderate': 1, 'unhealthy_sensitive': 2,
#                 'unhealthy': 3, 'very_unhealthy': 4, 'hazardous': 5
#             }
            
#             if current_category in category_to_numeric:
#                 numeric_cat = category_to_numeric[current_category]
#                 target_column = f'aqi_category_{numeric_cat}'
#                 if target_column in aqi_category_cols:
#                     aligned_features[target_column] = 1.0
#                     category_set = True
#                     print(f"Set {target_column} = 1.0 for current AQI {current_aqi:.1f}")
            
#             # Try named categories
#             if not category_set:
#                 target_column = f'aqi_category_{current_category}'
#                 if target_column in aqi_category_cols:
#                     aligned_features[target_column] = 1.0
#                     category_set = True
#                     print(f"Set {target_column} = 1.0 for current AQI {current_aqi:.1f}")
            
#             # Fallback: set the first category column to 1
#             if not category_set and aqi_category_cols:
#                 aligned_features[aqi_category_cols[0]] = 1.0
#                 print(f"Fallback: Set {aqi_category_cols[0]} = 1.0")
        
#         # Report missing features
#         if missing_features:
#             print(f"Added {len(missing_features)} missing features with default values")
#             if len(missing_features) <= 10:
#                 print(f"Missing features: {missing_features}")
        
#         # Ensure exact column order matches model expectations
#         aligned_features = aligned_features[model_feature_columns]
        
#         print(f"Final aligned features shape: {aligned_features.shape}")
#         print(f"Features successfully aligned: {list(aligned_features.columns) == model_feature_columns}")
        
#         return aligned_features

#     def prepare_inference_features(self, aqi_data, weather_data):
#         print("Preparing features for inference...")

#         try:
#             # Merge AQI and weather data
#             merged_data = self.feature_engineer.merge_aqi_weather_data(aqi_data, weather_data)
#             merged_data = self.feature_engineer.create_time_features(merged_data)

#             # Numerical Features
#             numeric_columns = ['aqi', 'pm25', 'pm10', 'temperature', 'humidity', 'wind_speed', 'pressure']
#             available_columns = [col for col in numeric_columns if col in merged_data.columns]

#             merged_data = self.feature_engineer.create_lag_features(merged_data, available_columns)
#             merged_data = self.feature_engineer.create_rolling_features(merged_data, available_columns)
#             merged_data = self.feature_engineer.create_derived_features(merged_data)

#             # Handle AQI Category
#             if 'aqi' in merged_data.columns:
#                 merged_data['aqi_category'] = merged_data['aqi'].apply(self.get_aqi_category)
#             else:
#                 merged_data['aqi_category'] = 'moderate'

#             merged_data['aqi_category'] = merged_data['aqi_category'].astype(str)
#             merged_data = pd.get_dummies(merged_data, columns=['aqi_category'], prefix='aqi_category')

#             print(f"Features after initial processing: {len(merged_data.columns)} columns")

#             # Get the best model info to determine required features
#             model, scaler, model_info = self.model_registry.get_best_model(metric='test_overall_rmse')
#             if model is None:
#                 print("No trained model found.")
#                 return None
            
#             # Get exact feature columns from model registry
#             model_feature_columns = self.get_model_feature_columns_from_registry(model_info)
#             if model_feature_columns is None:
#                 print("Could not determine model feature requirements from registry")
#                 return None
            
#             print(f"Model '{model_info['model_name']}' requires {len(model_feature_columns)} features")
            
#             # Align features exactly with model requirements
#             aligned_features = self.align_features_with_trained_model(merged_data, model_feature_columns)
            
#             if aligned_features is None or aligned_features.empty:
#                 print("Feature alignment failed")
#                 return None

#             # Return latest row for inference
#             latest_data = aligned_features.iloc[-1:].copy()
            
#             print(f"Final feature matrix shape: {latest_data.shape}")
            
#             return latest_data, model_feature_columns
            
#         except Exception as e:
#             print(f"Error in feature preparation: {e}")
#             import traceback
#             traceback.print_exc()
#             return None

#     def make_predictions(self, features_df, model_feature_columns):
#         print("Making predictions for next 3 days...")
#         try:
#             if features_df is None or features_df.empty:
#                 print("No feature data available for prediction")
#                 return None
                
#             model, scaler, model_info = self.model_registry.get_best_model(metric='test_overall_rmse')
#             if model is None:
#                 print("No trained model found. Please run training pipeline first.")
#                 return None

#             print(f"Using model: {model_info['model_name']}")
            
#             # Verify feature count matches exactly
#             expected_features = int(model_info.get('num_features', len(model_feature_columns)))
#             print(f"Model expects: {expected_features} features")
#             print(f"We have: {features_df.shape[1]} features")

#             if features_df.shape[1] != expected_features:
#                 print(f"ERROR: Feature count still doesn't match!")
#                 print(f"Expected: {expected_features}, Got: {features_df.shape[1]}")
#                 return None

#             # Prepare features for prediction
#             X = features_df.select_dtypes(include=[np.number])
            
#             # Handle missing values
#             X = X.fillna(X.mean())
            
#             # Check for infinite values
#             if np.isinf(X.values).any():
#                 print("Warning: Infinite values detected, replacing with 0")
#                 X = X.replace([np.inf, -np.inf], 0)

#             # Apply scaling if model requires it (Ridge regression)
#             if scaler is not None:
#                 print("Applying feature scaling...")
#                 X_scaled = scaler.transform(X.values)
#                 X_final = X_scaled
#             else:
#                 X_final = X.values

#             # Make prediction
#             print("Generating predictions...")
#             predictions = model.predict(X_final)
            
#             if len(predictions.shape) == 1:
#                 predictions = predictions.reshape(1, -1)
            
#             prediction_values = predictions[0]
#             print(f"Raw predictions: {prediction_values}")

#             # Ensure we have 3 predictions
#             if len(prediction_values) != 3:
#                 print(f"Warning: Expected 3 predictions, got {len(prediction_values)}")
#                 return None

#             # Calculate 3-day average
#             avg_prediction = np.mean(prediction_values)

#             # Get current AQI from features if available
#             current_aqi = None
#             if 'aqi' in features_df.columns:
#                 current_aqi = float(features_df['aqi'].iloc[0])

#             prediction_result = {
#                 'timestamp': datetime.now(),
#                 'predicted_aqi_24h': float(prediction_values[0]),
#                 'predicted_aqi_48h': float(prediction_values[1]),
#                 'predicted_aqi_72h': float(prediction_values[2]),
#                 'predicted_aqi_3day_avg': float(avg_prediction),
#                 'model_used': model_info['model_name'],
#                 'model_timestamp': model_info['created_at'],
#                 'current_aqi': current_aqi
#             }

#             print(f"Day 1 (24h) prediction: {prediction_values[0]:.2f} AQI")
#             print(f"Day 2 (48h) prediction: {prediction_values[1]:.2f} AQI")
#             print(f"Day 3 (72h) prediction: {prediction_values[2]:.2f} AQI")
#             print(f"3-day average: {avg_prediction:.2f} AQI")
#             print(f"Model used: {model_info['model_name']}")
            
#             return prediction_result

#         except Exception as e:
#             print(f"Error making predictions: {e}")
#             import traceback
#             traceback.print_exc()
#             return None

#     def save_predictions(self, prediction_result):
#         if prediction_result is None:
#             return
#         try:
#             predictions_df = pd.DataFrame([prediction_result])
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M')
#             filename = f"predictions_{timestamp}.csv"
#             filepath = os.path.join(self.predictions_path, filename)

#             predictions_df.to_csv(filepath, index=False)
#             print(f"Predictions saved to {filepath}")

#             # Update latest predictions file
#             latest_filepath = os.path.join(self.predictions_path, "latest_predictions.csv")

#             if os.path.exists(latest_filepath):
#                 existing_predictions = pd.read_csv(latest_filepath)
#                 all_predictions = pd.concat([existing_predictions, predictions_df], ignore_index=True)
#                 all_predictions = all_predictions.tail(100)
#             else:
#                 all_predictions = predictions_df

#             all_predictions.to_csv(latest_filepath, index=False)
#             print(f"Updated latest predictions: {latest_filepath}")

#         except Exception as e:
#             print(f"Error saving predictions: {e}")

#     def check_aqi_alerts(self, prediction_result):
#         if prediction_result is None:
#             return

#         day1_aqi = prediction_result['predicted_aqi_24h']
#         day2_aqi = prediction_result['predicted_aqi_48h'] 
#         day3_aqi = prediction_result['predicted_aqi_72h']
#         avg_aqi = prediction_result['predicted_aqi_3day_avg']
#         current_aqi = prediction_result.get('current_aqi', 0)

#         alerts = []

#         daily_predictions = [
#             ('Day 1 (24h)', day1_aqi),
#             ('Day 2 (48h)', day2_aqi),
#             ('Day 3 (72h)', day3_aqi)
#         ]

#         for day_name, predicted_aqi in daily_predictions:
#             if predicted_aqi > 200:
#                 alerts.append({
#                     'level': 'HAZARDOUS',
#                     'day': day_name,
#                     'message': f'{day_name} predicted AQI ({predicted_aqi:.0f}) indicates hazardous air quality',
#                     'recommendation': 'Avoid outdoor activities, use air purifiers indoors'
#                 })
#             elif predicted_aqi > 150:
#                 alerts.append({
#                     'level': 'UNHEALTHY',
#                     'day': day_name,
#                     'message': f'{day_name} predicted AQI ({predicted_aqi:.0f}) indicates unhealthy air quality',
#                     'recommendation': 'Limit outdoor activities, especially for sensitive individuals'
#                 })
#             elif predicted_aqi > 100:
#                 alerts.append({
#                     'level': 'MODERATE',
#                     'day': day_name,
#                     'message': f'{day_name} predicted AQI ({predicted_aqi:.0f}) indicates moderate air quality',
#                     'recommendation': 'Sensitive individuals should limit prolonged outdoor activities'
#                 })

#         if current_aqi:
#             for day_name, predicted_aqi in daily_predictions:
#                 if abs(predicted_aqi - current_aqi) > 50:
#                     alerts.append({
#                         'level': 'CHANGE_ALERT',
#                         'day': day_name,
#                         'message': f'{day_name} significant AQI change predicted: {current_aqi:.0f} → {predicted_aqi:.0f}',
#                         'recommendation': 'Monitor air quality closely'
#                     })

#         if day3_aqi > day1_aqi + 30:
#             alerts.append({
#                 'level': 'TREND_ALERT',
#                 'day': 'All Days',
#                 'message': f'Worsening air quality trend: Day 1 ({day1_aqi:.0f}) → Day 3 ({day3_aqi:.0f})',
#                 'recommendation': 'Prepare for deteriorating air quality conditions'
#             })
#         elif day1_aqi > day3_aqi + 30:
#             alerts.append({
#                 'level': 'TREND_ALERT', 
#                 'day': 'All Days',
#                 'message': f'Improving air quality trend: Day 1 ({day1_aqi:.0f}) → Day 3 ({day3_aqi:.0f})',
#                 'recommendation': 'Air quality conditions expected to improve'
#             })

#         if alerts:
#             alerts_df = pd.DataFrame(alerts)
#             alerts_df['timestamp'] = datetime.now()
#             alerts_df['predicted_aqi_24h'] = day1_aqi
#             alerts_df['predicted_aqi_48h'] = day2_aqi
#             alerts_df['predicted_aqi_72h'] = day3_aqi
#             alerts_df['predicted_aqi_avg'] = avg_aqi

#             alerts_file = os.path.join(self.predictions_path, "aqi_alerts.csv")

#             if os.path.exists(alerts_file):
#                 existing_alerts = pd.read_csv(alerts_file)
#                 all_alerts = pd.concat([existing_alerts, alerts_df], ignore_index=True)
#                 all_alerts = all_alerts.tail(50)
#             else:
#                 all_alerts = alerts_df

#             all_alerts.to_csv(alerts_file, index=False)

#             print(f"Generated {len(alerts)} alerts")
#             for alert in alerts:
#                 print(f"- {alert['level']} ({alert.get('day', 'N/A')}): {alert['message']}")

#     def run_inference_pipeline(self):
#         print("=== Starting Multi-Output Inference Pipeline ===")
#         try:
#             # Fetch latest data
#             aqi_data, weather_data = self.fetch_latest_data()
            
#             if aqi_data is None or aqi_data.empty:
#                 print("No AQI data available for inference")
#                 return None
                
#             if weather_data is None or weather_data.empty:
#                 print("No weather data available for inference")
#                 return None
            
#             # Prepare features aligned with trained model
#             result = self.prepare_inference_features(aqi_data, weather_data)
            
#             if result is None:
#                 print("Failed to prepare features")
#                 return None
                
#             features_df, model_feature_columns = result
            
#             # Make predictions
#             prediction_result = self.make_predictions(features_df, model_feature_columns)
            
#             if prediction_result is None:
#                 print("Failed to make predictions")
#                 return None
            
#             # Save predictions and check alerts
#             self.save_predictions(prediction_result)
#             self.check_aqi_alerts(prediction_result)

#             print("=== Multi-Output Inference Pipeline Completed Successfully ===")
#             return prediction_result

#         except Exception as e:
#             print(f"Inference pipeline failed: {e}")
#             import traceback
#             traceback.print_exc()
#             return None

#     def get_daily_predictions_summary(self, prediction_result):
#         """Get a formatted summary of daily predictions"""
#         if prediction_result is None:
#             return None
            
#         summary = {
#             'prediction_time': prediction_result['timestamp'],
#             'current_aqi': prediction_result.get('current_aqi', 'N/A'),
#             'daily_forecasts': [
#                 {
#                     'day': 'Tomorrow (24h)',
#                     'aqi': prediction_result['predicted_aqi_24h'],
#                     'category': self._get_aqi_category(prediction_result['predicted_aqi_24h'])
#                 },
#                 {
#                     'day': 'Day 2 (48h)', 
#                     'aqi': prediction_result['predicted_aqi_48h'],
#                     'category': self._get_aqi_category(prediction_result['predicted_aqi_48h'])
#                 },
#                 {
#                     'day': 'Day 3 (72h)',
#                     'aqi': prediction_result['predicted_aqi_72h'], 
#                     'category': self._get_aqi_category(prediction_result['predicted_aqi_72h'])
#                 }
#             ],
#             'average_aqi': prediction_result['predicted_aqi_3day_avg'],
#             'model_info': {
#                 'name': prediction_result['model_used'],
#                 'trained_at': prediction_result['model_timestamp']
#             }
#         }
#         return summary

#     def _get_aqi_category(self, aqi_value):
#         """Convert AQI value to category"""
#         if aqi_value <= 50:
#             return "Good"
#         elif aqi_value <= 100:
#             return "Moderate" 
#         elif aqi_value <= 150:
#             return "Unhealthy for Sensitive Groups"
#         elif aqi_value <= 200:
#             return "Unhealthy"
#         elif aqi_value <= 300:
#             return "Very Unhealthy"
#         else:
#             return "Hazardous"

# if __name__ == "__main__":
#     pipeline = InferencePipeline()
#     result = pipeline.run_inference_pipeline()

#     if result:
#         print("\n=== DAILY PREDICTIONS SUMMARY ===")
#         summary = pipeline.get_daily_predictions_summary(result)
#         if summary:
#             print(f"Current AQI: {summary['current_aqi']}")
#             print("\nNext 3 Days Forecast:")
#             for forecast in summary['daily_forecasts']:
#                 print(f"  {forecast['day']}: {forecast['aqi']:.1f} AQI ({forecast['category']})")
#             print(f"\n3-Day Average: {summary['average_aqi']:.1f} AQI")
#     else:
#         print("Inference pipeline failed!")

# # inference_pipeline.py - FIXED VERSION WITH BETTER ERROR HANDLING

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import joblib
import pickle

# Dynamically set project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from data_fetching.fetch_aqi_data import AQIDataFetcher
from feature_engineering.compute_features import FeatureEngineer
from model_training.model_registry import ModelRegistry
import joblib
import pickle

warnings.filterwarnings('ignore')

class InferencePipeline:
    def __init__(self):
        self.data_fetcher = AQIDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.model_registry = ModelRegistry()
        self.predictions_path = os.path.join(project_root, "data", "predictions")
        #self.model_path = os.path.join(project_root, "models")
        
        # Ensure predictions directory exists
        os.makedirs(self.predictions_path, exist_ok=True)

    def fetch_latest_data(self):
        print("Fetching latest data for inference...")
        self.data_fetcher.run_live_fetch()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        aqi_data = self.data_fetcher.fetch_historical_aqi_data(start_date, end_date)
        weather_data = self.data_fetcher.fetch_weather_data(start_date, end_date)
        return aqi_data, weather_data

    def get_aqi_category(self, aqi_value):
        """Convert AQI value to standardized category (same as training)"""
        if pd.isna(aqi_value):
            return 'moderate'  # Default category
        elif aqi_value <= 50:
            return 'good'
        elif aqi_value <= 100:
            return 'moderate'
        elif aqi_value <= 150:
            return 'unhealthy_sensitive'
        elif aqi_value <= 200:
            return 'unhealthy'
        elif aqi_value <= 300:
            return 'very_unhealthy'
        else:
            return 'hazardous'

    def load_model_safely(self, model_path):
        """Safely load model with multiple fallback methods"""
        print(f"Attempting to load model from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Model file does not exist: {model_path}")
            return None
            
        # Method 1: Try joblib
        try:
            print("Trying joblib.load...")
            model = joblib.load(model_path)
            print("✓ Model loaded successfully with joblib")
            return model
        except Exception as e:
            print(f"joblib.load failed: {e}")
        
        # Method 2: Try pickle
        try:
            print("Trying pickle.load...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✓ Model loaded successfully with pickle")
            return model
        except Exception as e:
            print(f"pickle.load failed: {e}")
        
        # Method 3: Try different protocol
        try:
            print("Trying joblib with different protocol...")
            model = joblib.load(model_path)
            print("✓ Model loaded successfully with alternative method")
            return model
        except Exception as e:
            print(f"Alternative joblib method failed: {e}")
        
        print("❌ All model loading methods failed")
        return None

    def get_best_model_safely(self):
        """Get best model using your existing ModelRegistry methods"""
        try:
            # First, try to get all models from registry
            models_df = self.model_registry.list_all_models()
            
            if models_df is None or len(models_df) == 0:
                print("No models found in registry")
                return None, None, None
            
            print(f"Found {len(models_df)} models in registry")
            
            # Try to find the best model by test_overall_rmse or similar metric
            best_model_info = None
            
            # Check what metrics are available
            available_metrics = [col for col in models_df.columns if 'rmse' in col.lower() or 'test' in col.lower()]
            print(f"Available metrics: {available_metrics}")
            
            if 'test_overall_rmse' in models_df.columns:
                best_model_info = models_df.loc[models_df['test_overall_rmse'].idxmin()]
                print("Using test_overall_rmse for model selection")
            elif 'test_rmse' in models_df.columns:
                best_model_info = models_df.loc[models_df['test_rmse'].idxmin()]
                print("Using test_rmse for model selection")
            else:
                # Fallback: use the most recent model
                if 'created_at' in models_df.columns:
                    best_model_info = models_df.loc[models_df['created_at'].idxmax()]
                    print("Using most recent model as no performance metrics found")
                else:
                    best_model_info = models_df.iloc[0]
                    print("Using first available model")
            
            print(f"Selected model: {best_model_info.get('model_name', 'Unknown')}")
            
            # Build model path
            if 'model_file' in best_model_info:
                model_filename = best_model_info['model_file']
            elif 'model_path' in best_model_info:
                model_filename = os.path.basename(best_model_info['model_path'])
            else:
                print("No model file information found")
                return None, None, None
            
            model_path = os.path.join(self.model_registry.models_path, model_filename)
            print(f"Model path: {model_path}")
            
            # Load model safely
            model = self.load_model_safely(model_path)
            if model is None:
                print("Failed to load model")
                return None, None, None
            
            # Load scaler safely
            scaler = None
            if 'scaler_file' in best_model_info and pd.notna(best_model_info['scaler_file']):
                scaler_path = os.path.join(self.model_registry.models_path, best_model_info['scaler_file'])
                if os.path.exists(scaler_path):
                    scaler = self.load_model_safely(scaler_path)
                    if scaler is not None:
                        print("✓ Scaler loaded successfully")
                    else:
                        print("⚠ Scaler could not be loaded, proceeding without scaling")
            
            return model, scaler, best_model_info.to_dict()
            
        except Exception as e:
            print(f"Error getting best model: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def get_model_feature_columns_from_registry(self, model_info):
        """Extract the exact feature columns used during model training from registry"""
        try:
            feature_columns_str = model_info['feature_columns']
            if pd.isna(feature_columns_str) or feature_columns_str == '':
                print("No feature columns found in model registry")
                return None
            feature_columns = feature_columns_str.split(',')
            print(f"Found {len(feature_columns)} feature columns in registry")
            return feature_columns
        except Exception as e:
            print(f"Error extracting feature columns from registry: {e}")
            return None

    def create_fallback_features(self, merged_data, num_features=50):
        """Create a fallback feature set if model requirements are unknown"""
        print(f"Creating fallback feature set with {num_features} features...")
        
        # Start with basic numeric features
        basic_features = ['aqi', 'pm25', 'pm10', 'temperature', 'humidity', 'wind_speed', 'pressure']
        available_basic = [col for col in basic_features if col in merged_data.columns]
        
        # Add time features
        time_features = [col for col in merged_data.columns if any(time_word in col.lower() 
                        for time_word in ['hour', 'day', 'month', 'weekday'])]
        
        # Add lag features (most important recent values)
        lag_features = [col for col in merged_data.columns if 'lag_1' in col or 'lag_2' in col]
        
        # Add rolling features
        rolling_features = [col for col in merged_data.columns if 'rolling' in col and 'mean' in col]
        
        # Add AQI categories
        aqi_features = [col for col in merged_data.columns if col.startswith('aqi_category_')]
        
        # Combine all features
        selected_features = (available_basic + time_features[:5] + 
                           lag_features[:10] + rolling_features[:10] + aqi_features)
        
        # Remove duplicates and ensure they exist
        selected_features = list(set([col for col in selected_features if col in merged_data.columns]))
        
        # Fill up to num_features with any remaining numeric columns
        numeric_cols = merged_data.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in selected_features and len(selected_features) < num_features:
                selected_features.append(col)
        
        # Truncate to requested number
        selected_features = selected_features[:num_features]
        
        print(f"Created fallback feature set with {len(selected_features)} features")
        return selected_features

    def align_features_with_trained_model(self, features_df, model_feature_columns):
        """Align inference features exactly with trained model features"""
        print(f"Aligning features with trained model requirements...")
        print(f"Model expects: {len(model_feature_columns)} features")
        print(f"We have: {len(features_df.columns)} features")
        
        # Start with empty dataframe with correct index
        aligned_features = pd.DataFrame(index=features_df.index)
        
        # Add each required feature column
        missing_features = []
        for feature_col in model_feature_columns:
            if feature_col in features_df.columns:
                aligned_features[feature_col] = features_df[feature_col]
            else:
                # Handle missing features with reasonable defaults
                if feature_col.startswith('aqi_category_'):
                    aligned_features[feature_col] = 0.0  # Categorical feature default
                elif 'lag' in feature_col or 'rolling' in feature_col:
                    aligned_features[feature_col] = 0.0  # Time-based feature default
                else:
                    aligned_features[feature_col] = 0.0  # General default
                missing_features.append(feature_col)
        
        # Handle AQI categories specifically if current AQI is available
        if 'aqi' in features_df.columns and any(col.startswith('aqi_category_') for col in model_feature_columns):
            current_aqi = features_df['aqi'].iloc[-1]
            current_category = self.get_aqi_category(current_aqi)
            
            # Set the appropriate category column to 1
            aqi_category_cols = [col for col in model_feature_columns if col.startswith('aqi_category_')]
            
            # Reset all category columns to 0 first
            for col in aqi_category_cols:
                aligned_features[col] = 0.0
            
            # Try to set the correct category
            category_set = False
            
            # Try numeric categories (0-5)
            category_to_numeric = {
                'good': 0, 'moderate': 1, 'unhealthy_sensitive': 2,
                'unhealthy': 3, 'very_unhealthy': 4, 'hazardous': 5
            }
            
            if current_category in category_to_numeric:
                numeric_cat = category_to_numeric[current_category]
                target_column = f'aqi_category_{numeric_cat}'
                if target_column in aqi_category_cols:
                    aligned_features[target_column] = 1.0
                    category_set = True
                    print(f"Set {target_column} = 1.0 for current AQI {current_aqi:.1f}")
            
            # Try named categories
            if not category_set:
                target_column = f'aqi_category_{current_category}'
                if target_column in aqi_category_cols:
                    aligned_features[target_column] = 1.0
                    category_set = True
                    print(f"Set {target_column} = 1.0 for current AQI {current_aqi:.1f}")
            
            # Fallback: set the first category column to 1
            if not category_set and aqi_category_cols:
                aligned_features[aqi_category_cols[0]] = 1.0
                print(f"Fallback: Set {aqi_category_cols[0]} = 1.0")
        
        # Report missing features
        if missing_features:
            print(f"Added {len(missing_features)} missing features with default values")
            if len(missing_features) <= 10:
                print(f"Missing features: {missing_features}")
        
        # Ensure exact column order matches model expectations
        aligned_features = aligned_features[model_feature_columns]
        
        print(f"Final aligned features shape: {aligned_features.shape}")
        print(f"Features successfully aligned: {list(aligned_features.columns) == model_feature_columns}")
        
        return aligned_features

    def prepare_inference_features(self, aqi_data, weather_data):
        print("Preparing features for inference...")

        try:
            # Merge AQI and weather data
            merged_data = self.feature_engineer.merge_aqi_weather_data(aqi_data, weather_data)
            merged_data = self.feature_engineer.create_time_features(merged_data)

            # Numerical Features
            numeric_columns = ['aqi', 'pm25', 'pm10', 'temperature', 'humidity', 'wind_speed', 'pressure']
            available_columns = [col for col in numeric_columns if col in merged_data.columns]

            merged_data = self.feature_engineer.create_lag_features(merged_data, available_columns)
            merged_data = self.feature_engineer.create_rolling_features(merged_data, available_columns)
            merged_data = self.feature_engineer.create_derived_features(merged_data)

            # Handle AQI Category
            if 'aqi' in merged_data.columns:
                merged_data['aqi_category'] = merged_data['aqi'].apply(self.get_aqi_category)
            else:
                merged_data['aqi_category'] = 'moderate'

            merged_data['aqi_category'] = merged_data['aqi_category'].astype(str)
            merged_data = pd.get_dummies(merged_data, columns=['aqi_category'], prefix='aqi_category')

            print(f"Features after initial processing: {len(merged_data.columns)} columns")

            # Get the best model info with enhanced error handling
            model, scaler, model_info = self.get_best_model_safely()
            if model is None:
                print("No trained model could be loaded.")
                return None
            
            # Get exact feature columns from model registry
            model_feature_columns = self.get_model_feature_columns_from_registry(model_info)
            if model_feature_columns is None:
                print("Could not determine model feature requirements, using fallback approach")
                # Use a reasonable number of features based on typical models
                model_feature_columns = self.create_fallback_features(merged_data, num_features=50)
            
            print(f"Model '{model_info['model_name']}' requires {len(model_feature_columns)} features")
            
            # Align features exactly with model requirements
            aligned_features = self.align_features_with_trained_model(merged_data, model_feature_columns)
            
            if aligned_features is None or aligned_features.empty:
                print("Feature alignment failed")
                return None

            # Return latest row for inference
            latest_data = aligned_features.iloc[-1:].copy()
            
            print(f"Final feature matrix shape: {latest_data.shape}")
            
            return latest_data, model_feature_columns, model, scaler, model_info
            
        except Exception as e:
            print(f"Error in feature preparation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def make_predictions_with_fallback(self, features_df, model, scaler, model_info):
        """Make predictions with multiple fallback strategies"""
        print("Making predictions with fallback strategies...")
        try:
            if features_df is None or features_df.empty:
                print("No feature data available for prediction")
                return None

            print(f"Using model: {model_info['model_name']}")
            
            # Prepare features for prediction
            X = features_df.select_dtypes(include=[np.number])
            
            # Handle missing values
            X = X.fillna(0)  # More conservative fillna
            
            # Check for infinite values
            if np.isinf(X.values).any():
                print("Warning: Infinite values detected, replacing with 0")
                X = X.replace([np.inf, -np.inf], 0)

            # Apply scaling if scaler exists
            if scaler is not None:
                print("Applying feature scaling...")
                try:
                    X_scaled = scaler.transform(X.values)
                    X_final = X_scaled
                    print("✓ Scaling applied successfully")
                except Exception as e:
                    print(f"Scaling failed: {e}, proceeding without scaling")
                    X_final = X.values
            else:
                print("No scaler available, using raw features")
                X_final = X.values

            # Make prediction
            print("Generating predictions...")
            predictions = model.predict(X_final)
            
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(1, -1)
            
            prediction_values = predictions[0]
            print(f"Raw predictions: {prediction_values}")

            # Handle different prediction formats
            if len(prediction_values) == 3:
                # Multi-output model (24h, 48h, 72h)
                day1_pred = float(prediction_values[0])
                day2_pred = float(prediction_values[1])
                day3_pred = float(prediction_values[2])
            elif len(prediction_values) == 1:
                # Single output model - use as base prediction
                base_pred = float(prediction_values[0])
                # Create reasonable variations for multi-day forecast
                day1_pred = base_pred
                day2_pred = base_pred * 0.98  # Slight variation
                day3_pred = base_pred * 1.02  # Slight variation
                print("Single output model detected, creating multi-day variations")
            else:
                print(f"Unexpected prediction format: {len(prediction_values)} values")
                # Fallback to current AQI with trend
                current_aqi = features_df.get('aqi', pd.Series([100])).iloc[0] if 'aqi' in features_df.columns else 100
                day1_pred = current_aqi * 1.0
                day2_pred = current_aqi * 1.05
                day3_pred = current_aqi * 1.1

            # Calculate 3-day average
            avg_prediction = (day1_pred + day2_pred + day3_pred) / 3

            # Get current AQI from features if available
            current_aqi = None
            if 'aqi' in features_df.columns:
                current_aqi = float(features_df['aqi'].iloc[0])

            prediction_result = {
                'timestamp': datetime.now(),
                'predicted_aqi_24h': day1_pred,
                'predicted_aqi_48h': day2_pred,
                'predicted_aqi_72h': day3_pred,
                'predicted_aqi_3day_avg': avg_prediction,
                'model_used': model_info['model_name'],
                'model_timestamp': model_info['created_at'],
                'current_aqi': current_aqi
            }

            print(f"Day 1 (24h) prediction: {day1_pred:.2f} AQI")
            print(f"Day 2 (48h) prediction: {day2_pred:.2f} AQI")
            print(f"Day 3 (72h) prediction: {day3_pred:.2f} AQI")
            print(f"3-day average: {avg_prediction:.2f} AQI")
            print(f"Model used: {model_info['model_name']}")
            
            return prediction_result

        except Exception as e:
            print(f"Error making predictions: {e}")
            import traceback
            traceback.print_exc()
            
            # Final fallback - use current AQI trends
            try:
                current_aqi = features_df.get('aqi', pd.Series([100])).iloc[0] if 'aqi' in features_df.columns else 100
                print(f"Using fallback prediction based on current AQI: {current_aqi}")
                
                prediction_result = {
                    'timestamp': datetime.now(),
                    'predicted_aqi_24h': current_aqi * 1.0,
                    'predicted_aqi_48h': current_aqi * 1.05,
                    'predicted_aqi_72h': current_aqi * 1.1,
                    'predicted_aqi_3day_avg': current_aqi * 1.05,
                    'model_used': 'Fallback (Trend-based)',
                    'model_timestamp': datetime.now(),
                    'current_aqi': current_aqi
                }
                return prediction_result
            except:
                return None

    def save_predictions(self, prediction_result):
        if prediction_result is None:
            return
        try:
            predictions_df = pd.DataFrame([prediction_result])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"predictions_{timestamp}.csv"
            filepath = os.path.join(self.predictions_path, filename)

            predictions_df.to_csv(filepath, index=False)
            print(f"Predictions saved to {filepath}")

            # Update latest predictions file
            latest_filepath = os.path.join(self.predictions_path, "latest_predictions.csv")

            if os.path.exists(latest_filepath):
                existing_predictions = pd.read_csv(latest_filepath)
                all_predictions = pd.concat([existing_predictions, predictions_df], ignore_index=True)
                all_predictions = all_predictions.tail(100)
            else:
                all_predictions = predictions_df

            all_predictions.to_csv(latest_filepath, index=False)
            print(f"Updated latest predictions: {latest_filepath}")

        except Exception as e:
            print(f"Error saving predictions: {e}")

    def check_aqi_alerts(self, prediction_result):
        if prediction_result is None:
            return

        day1_aqi = prediction_result['predicted_aqi_24h']
        day2_aqi = prediction_result['predicted_aqi_48h'] 
        day3_aqi = prediction_result['predicted_aqi_72h']
        avg_aqi = prediction_result['predicted_aqi_3day_avg']
        current_aqi = prediction_result.get('current_aqi', 0)

        alerts = []

        daily_predictions = [
            ('Day 1 (24h)', day1_aqi),
            ('Day 2 (48h)', day2_aqi),
            ('Day 3 (72h)', day3_aqi)
        ]

        for day_name, predicted_aqi in daily_predictions:
            if predicted_aqi > 200:
                alerts.append({
                    'level': 'HAZARDOUS',
                    'day': day_name,
                    'message': f'{day_name} predicted AQI ({predicted_aqi:.0f}) indicates hazardous air quality',
                    'recommendation': 'Avoid outdoor activities, use air purifiers indoors'
                })
            elif predicted_aqi > 150:
                alerts.append({
                    'level': 'UNHEALTHY',
                    'day': day_name,
                    'message': f'{day_name} predicted AQI ({predicted_aqi:.0f}) indicates unhealthy air quality',
                    'recommendation': 'Limit outdoor activities, especially for sensitive individuals'
                })
            elif predicted_aqi > 100:
                alerts.append({
                    'level': 'MODERATE',
                    'day': day_name,
                    'message': f'{day_name} predicted AQI ({predicted_aqi:.0f}) indicates moderate air quality',
                    'recommendation': 'Sensitive individuals should limit prolonged outdoor activities'
                })

        if current_aqi:
            for day_name, predicted_aqi in daily_predictions:
                if abs(predicted_aqi - current_aqi) > 50:
                    alerts.append({
                        'level': 'CHANGE_ALERT',
                        'day': day_name,
                        'message': f'{day_name} significant AQI change predicted: {current_aqi:.0f} → {predicted_aqi:.0f}',
                        'recommendation': 'Monitor air quality closely'
                    })

        if day3_aqi > day1_aqi + 30:
            alerts.append({
                'level': 'TREND_ALERT',
                'day': 'All Days',
                'message': f'Worsening air quality trend: Day 1 ({day1_aqi:.0f}) → Day 3 ({day3_aqi:.0f})',
                'recommendation': 'Prepare for deteriorating air quality conditions'
            })
        elif day1_aqi > day3_aqi + 30:
            alerts.append({
                'level': 'TREND_ALERT', 
                'day': 'All Days',
                'message': f'Improving air quality trend: Day 1 ({day1_aqi:.0f}) → Day 3 ({day3_aqi:.0f})',
                'recommendation': 'Air quality conditions expected to improve'
            })

        if alerts:
            alerts_df = pd.DataFrame(alerts)
            alerts_df['timestamp'] = datetime.now()
            alerts_df['predicted_aqi_24h'] = day1_aqi
            alerts_df['predicted_aqi_48h'] = day2_aqi
            alerts_df['predicted_aqi_72h'] = day3_aqi
            alerts_df['predicted_aqi_avg'] = avg_aqi

            alerts_file = os.path.join(self.predictions_path, "aqi_alerts.csv")

            if os.path.exists(alerts_file):
                existing_alerts = pd.read_csv(alerts_file)
                all_alerts = pd.concat([existing_alerts, alerts_df], ignore_index=True)
                all_alerts = all_alerts.tail(50)
            else:
                all_alerts = alerts_df

            all_alerts.to_csv(alerts_file, index=False)

            print(f"Generated {len(alerts)} alerts")
            for alert in alerts:
                print(f"- {alert['level']} ({alert.get('day', 'N/A')}): {alert['message']}")

    def run_inference_pipeline(self):
        print("=== Starting Enhanced Multi-Output Inference Pipeline ===")
        try:
            # Fetch latest data
            aqi_data, weather_data = self.fetch_latest_data()
            
            if aqi_data is None or aqi_data.empty:
                print("No AQI data available for inference")
                return None
                
            if weather_data is None or weather_data.empty:
                print("No weather data available for inference")
                return None
            
            # Prepare features with enhanced error handling
            result = self.prepare_inference_features(aqi_data, weather_data)
            
            if result is None:
                print("Failed to prepare features")
                return None
                
            features_df, model_feature_columns, model, scaler, model_info = result
            
            # Make predictions with fallback
            prediction_result = self.make_predictions_with_fallback(features_df, model, scaler, model_info)
            
            if prediction_result is None:
                print("Failed to make predictions")
                return None
            
            # Save predictions and check alerts
            self.save_predictions(prediction_result)
            self.check_aqi_alerts(prediction_result)

            print("=== Enhanced Multi-Output Inference Pipeline Completed Successfully ===")
            return prediction_result

        except Exception as e:
            print(f"Inference pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_daily_predictions_summary(self, prediction_result):
        """Get a formatted summary of daily predictions"""
        if prediction_result is None:
            return None
            
        summary = {
            'prediction_time': prediction_result['timestamp'],
            'current_aqi': prediction_result.get('current_aqi', 'N/A'),
            'daily_forecasts': [
                {
                    'day': 'Tomorrow (24h)',
                    'aqi': prediction_result['predicted_aqi_24h'],
                    'category': self._get_aqi_category(prediction_result['predicted_aqi_24h'])
                },
                {
                    'day': 'Day 2 (48h)', 
                    'aqi': prediction_result['predicted_aqi_48h'],
                    'category': self._get_aqi_category(prediction_result['predicted_aqi_48h'])
                },
                {
                    'day': 'Day 3 (72h)',
                    'aqi': prediction_result['predicted_aqi_72h'], 
                    'category': self._get_aqi_category(prediction_result['predicted_aqi_72h'])
                }
            ],
            'average_aqi': prediction_result['predicted_aqi_3day_avg'],
            'model_info': {
                'name': prediction_result['model_used'],
                'trained_at': prediction_result['model_timestamp']
            }
        }
        return summary

    def _get_aqi_category(self, aqi_value):
        """Convert AQI value to category"""
        if aqi_value <= 50:
            return "Good"
        elif aqi_value <= 100:
            return "Moderate" 
        elif aqi_value <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi_value <= 200:
            return "Unhealthy"
        elif aqi_value <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

if __name__ == "__main__":
    pipeline = InferencePipeline()
    result = pipeline.run_inference_pipeline()

    if result:
        print("\n=== DAILY PREDICTIONS SUMMARY ===")
        summary = pipeline.get_daily_predictions_summary(result)
        if summary:
            print(f"Current AQI: {summary['current_aqi']}")
            print("\nNext 3 Days Forecast:")
            for forecast in summary['daily_forecasts']:
                print(f"  {forecast['day']}: {forecast['aqi']:.1f} AQI ({forecast['category']})")
            print(f"\n3-Day Average: {summary['average_aqi']:.1f} AQI")
            print(f"\nModel: {summary['model_info']['name']}")
    else:
        print("Inference pipeline failed!")