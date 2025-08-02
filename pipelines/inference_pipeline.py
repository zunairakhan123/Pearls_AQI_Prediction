# import sys
# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import warnings

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
#         self.predictions_path = os.path.join(project_root, "data/predictions/")

#     def fetch_latest_data(self):
#         print("Fetching latest data for inference...")
#         self.data_fetcher.run_live_fetch()
#         end_date = datetime.now()
#         start_date = end_date - timedelta(days=7)
#         aqi_data = self.data_fetcher.fetch_historical_aqi_data(start_date, end_date)
#         weather_data = self.data_fetcher.fetch_weather_data(start_date, end_date)
#         return aqi_data, weather_data

#     def prepare_inference_features(self, aqi_data, weather_data, expected_feature_columns):
#         print("Preparing features for inference...")

#         # Merge AQI and weather data
#         merged_data = self.feature_engineer.merge_aqi_weather_data(aqi_data, weather_data)
#         merged_data = self.feature_engineer.create_time_features(merged_data)

#         # Numerical Features
#         numeric_columns = ['aqi', 'pm25', 'pm10', 'temperature', 'humidity', 'wind_speed', 'pressure']
#         available_columns = [col for col in numeric_columns if col in merged_data.columns]

#         merged_data = self.feature_engineer.create_lag_features(merged_data, available_columns)
#         merged_data = self.feature_engineer.create_rolling_features(merged_data, available_columns)
#         merged_data = self.feature_engineer.create_derived_features(merged_data)

#         # --- One-Hot Encode aqi_category ---
#         merged_data['aqi_category'] = merged_data['aqi_category'].astype(str)
#         merged_data = pd.get_dummies(merged_data, columns=['aqi_category'], prefix='aqi_category')

#         # Ensure all expected dummy columns exist
#         for col in expected_feature_columns:
#             if col not in merged_data.columns:
#                 merged_data[col] = 0  # Add missing dummy column

#         # Drop any extra columns not in expected
#         extra_cols = set(merged_data.columns) - set(expected_feature_columns)
#         merged_data.drop(columns=extra_cols, inplace=True, errors='ignore')

#         # Ensure column order is exactly same
#         merged_data = merged_data[expected_feature_columns]

#         # Return latest row for inference
#         latest_data = merged_data.iloc[-1:].copy()
#         return latest_data

#     def make_predictions(self, features_df):
#         print("Making predictions...")
#         try:
#             model, scaler, model_info = self.model_registry.get_best_model(metric='test_rmse')
#             if model is None:
#                 print("No trained model found. Please run training pipeline first.")
#                 return None

#             X = features_df.select_dtypes(include=[np.number])
#             X = X.fillna(X.mean())

#             if scaler is not None:
#                 X = scaler.transform(X)

#             prediction = model.predict(X)[0]

#             prediction_result = {
#                 'timestamp': datetime.now(),
#                 'predicted_aqi_3day_avg': prediction,
#                 'model_used': model_info['model_name'],
#                 'model_timestamp': model_info['created_at'],
#                 'current_aqi': features_df['aqi'].iloc[0] if 'aqi' in features_df.columns else None
#             }

#             prediction_result['predicted_aqi_24h'] = prediction * 0.95
#             prediction_result['predicted_aqi_48h'] = prediction
#             prediction_result['predicted_aqi_72h'] = prediction * 1.05

#             print(f"Prediction made: {prediction:.2f} AQI (3-day average)")
#             print(f"Model used: {model_info['model_name']}")
#             return prediction_result

#         except Exception as e:
#             print(f"Error making predictions: {e}")
#             return None

#     def save_predictions(self, prediction_result):
#         if prediction_result is None:
#             return
#         try:
#             predictions_df = pd.DataFrame([prediction_result])
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M')
#             filename = f"predictions_{timestamp}.csv"
#             filepath = os.path.join(self.predictions_path, filename)

#             os.makedirs(self.predictions_path, exist_ok=True)
#             predictions_df.to_csv(filepath, index=False)
#             print(f"Predictions saved to {filepath}")

#             latest_filepath = os.path.join(self.predictions_path, "latest_predictions.csv")

#             if os.path.exists(latest_filepath):
#                 existing_predictions = pd.read_csv(latest_filepath)
#                 all_predictions = pd.concat([existing_predictions, predictions_df], ignore_index=True)
#                 all_predictions = all_predictions.tail(100)
#             else:
#                 all_predictions = predictions_df

#             all_predictions.to_csv(latest_filepath, index=False)

#         except Exception as e:
#             print(f"Error saving predictions: {e}")

#     def check_aqi_alerts(self, prediction_result):
#         if prediction_result is None:
#             return

#         predicted_aqi = prediction_result['predicted_aqi_3day_avg']
#         current_aqi = prediction_result.get('current_aqi', 0)

#         alerts = []

#         if predicted_aqi > 200:
#             alerts.append({
#                 'level': 'HAZARDOUS',
#                 'message': f'Predicted AQI ({predicted_aqi:.0f}) indicates hazardous air quality',
#                 'recommendation': 'Avoid outdoor activities, use air purifiers indoors'
#             })
#         elif predicted_aqi > 150:
#             alerts.append({
#                 'level': 'UNHEALTHY',
#                 'message': f'Predicted AQI ({predicted_aqi:.0f}) indicates unhealthy air quality',
#                 'recommendation': 'Limit outdoor activities, especially for sensitive individuals'
#             })
#         elif predicted_aqi > 100:
#             alerts.append({
#                 'level': 'MODERATE',
#                 'message': f'Predicted AQI ({predicted_aqi:.0f}) indicates moderate air quality',
#                 'recommendation': 'Sensitive individuals should limit prolonged outdoor activities'
#             })

#         if current_aqi and abs(predicted_aqi - current_aqi) > 50:
#             alerts.append({
#                 'level': 'CHANGE_ALERT',
#                 'message': f'Significant AQI change predicted: {current_aqi:.0f} → {predicted_aqi:.0f}',
#                 'recommendation': 'Monitor air quality closely'
#             })

#         if alerts:
#             alerts_df = pd.DataFrame(alerts)
#             alerts_df['timestamp'] = datetime.now()
#             alerts_df['predicted_aqi'] = predicted_aqi

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
#                 print(f"- {alert['level']}: {alert['message']}")

#     def run_inference_pipeline(self):
#         print("=== Starting Inference Pipeline ===")
#         try:
#             aqi_data, weather_data = self.fetch_latest_data()
#             model, _, model_info = self.model_registry.get_best_model(metric='test_rmse')
#             expected_feature_columns = model_info['feature_columns'].split(',')

#             features_df = self.prepare_inference_features(aqi_data, weather_data, expected_feature_columns)
#             prediction_result = self.make_predictions(features_df)
#             self.save_predictions(prediction_result)
#             self.check_aqi_alerts(prediction_result)

#             print("=== Inference Pipeline Completed ===")
#             return prediction_result

#         except Exception as e:
#             print(f"Inference pipeline failed: {e}")
#             return None

# if __name__ == "__main__":
#     pipeline = InferencePipeline()
#     result = pipeline.run_inference_pipeline()

#     if result:
#         print(f"Prediction: {result['predicted_aqi_3day_avg']:.2f} AQI")
#     else:
#         print("Inference pipeline failed!"

#inference_pipeline.py

#inference_pipeline.py

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import joblib

# Dynamically set project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from data_fetching.fetch_aqi_data import AQIDataFetcher
from feature_engineering.compute_features import FeatureEngineer
from model_training.model_registry import ModelRegistry

warnings.filterwarnings('ignore')

class InferencePipeline:
    def __init__(self):
        self.data_fetcher = AQIDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.model_registry = ModelRegistry()
        self.predictions_path = os.path.join(project_root, "data", "predictions")
        
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

    def get_actual_model_feature_count(self, model):
        """Get the actual number of features the trained model expects"""
        try:
            # For MultiOutputRegressor, check the first estimator
            if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                first_estimator = model.estimators_[0]
                if hasattr(first_estimator, 'n_features_in_'):
                    return first_estimator.n_features_in_
            
            # Fallback: check if model has n_features_in_ directly
            if hasattr(model, 'n_features_in_'):
                return model.n_features_in_
                
            return None
        except:
            return None

    def load_actual_training_features(self):
        """Load the actual training features used to train the model"""
        try:
            # Load from feature store to get the actual training features
            from feature_store.feature_store_manager import FeatureStore
            fs = FeatureStore()
            training_features_df = fs.load_features_from_csv(latest=True)
            
            if training_features_df is None:
                print("Warning: Could not load training features")
                return None, None
            
            # Get the columns used for training (excluding targets and metadata)
            exclude_columns = ['timestamp', 'city', 'country', 'target_aqi_24h', 
                              'target_aqi_48h', 'target_aqi_72h', 'target_aqi_3day_avg']
            
            feature_columns = [col for col in training_features_df.columns if col not in exclude_columns]
            training_X = training_features_df[feature_columns].select_dtypes(include=[np.number])
            
            # Extract AQI category columns
            aqi_category_columns = [col for col in training_X.columns if col.startswith('aqi_category_')]
            
            print(f"Actual training features loaded: {len(training_X.columns)} features")
            print(f"Actual AQI category columns: {aqi_category_columns}")
            
            return list(training_X.columns), aqi_category_columns
            
        except Exception as e:
            print(f"Could not load actual training features: {e}")
            return None, None

    def align_features_with_actual_training(self, features_df, actual_training_columns, actual_aqi_categories):
        """Align features with the actual training data (not registry info)"""
        print("Aligning features with actual training data...")
        
        # Start with a copy
        aligned_features = features_df.copy()
        
        print(f"Original inference features: {len(aligned_features.columns)}")
        print(f"Actual training features needed: {len(actual_training_columns)}")
        
        # Handle AQI category columns specifically
        current_aqi_categories = [col for col in aligned_features.columns if col.startswith('aqi_category_')]
        print(f"Current AQI categories: {current_aqi_categories}")
        print(f"Actual training AQI categories: {actual_aqi_categories}")
        
        # Remove current AQI category columns first
        aligned_features = aligned_features.drop(columns=current_aqi_categories, errors='ignore')
        
        # Add all actual training AQI categories with default value 0
        for aqi_cat_col in actual_aqi_categories:
            aligned_features[aqi_cat_col] = 0.0
        
        # Set the appropriate category to 1 based on current AQI value
        if 'aqi' in features_df.columns:
            current_aqi = features_df['aqi'].iloc[-1]  # Get the latest AQI value
            current_category = self.get_aqi_category(current_aqi)
            
            # Find which actual category column corresponds to current category
            # Check if the actual training data uses numeric categories (like aqi_category_5)
            # or named categories (like aqi_category_moderate)
            category_set = False
            
            # First, try numeric categories (0-5)
            category_to_numeric = {
                'good': 0, 'moderate': 1, 'unhealthy_sensitive': 2,
                'unhealthy': 3, 'very_unhealthy': 4, 'hazardous': 5
            }
            
            if current_category in category_to_numeric:
                numeric_cat = category_to_numeric[current_category]
                target_column = f'aqi_category_{numeric_cat}'
                if target_column in actual_aqi_categories:
                    aligned_features[target_column] = 1.0
                    print(f"Set {target_column} = 1.0 for AQI category '{current_category}' (AQI: {current_aqi:.1f})")
                    category_set = True
            
            # If numeric didn't work, try named categories
            if not category_set:
                target_column = f'aqi_category_{current_category}'
                if target_column in actual_aqi_categories:
                    aligned_features[target_column] = 1.0
                    print(f"Set {target_column} = 1.0 for AQI category '{current_category}' (AQI: {current_aqi:.1f})")
                    category_set = True
            
            # Fallback: set the first available category column to 1
            if not category_set and actual_aqi_categories:
                aligned_features[actual_aqi_categories[0]] = 1.0
                print(f"Fallback: Set {actual_aqi_categories[0]} = 1.0")
        
        # Add other missing columns with default values
        missing_cols = []
        for col in actual_training_columns:
            if col not in aligned_features.columns:
                aligned_features[col] = 0.0  # Default value for missing features
                missing_cols.append(col)
        
        if missing_cols:
            print(f"Added {len(missing_cols)} missing columns with default values")
            
        # Remove extra columns not in actual training
        extra_cols = []
        for col in aligned_features.columns:
            if col not in actual_training_columns:
                extra_cols.append(col)
        
        if extra_cols:
            print(f"Removing {len(extra_cols)} extra columns not in actual training")
            aligned_features = aligned_features.drop(columns=extra_cols)
        
        # Reorder columns to match actual training exactly
        aligned_features = aligned_features[actual_training_columns]
        
        print(f"Final aligned features: {len(aligned_features.columns)}")
        
        # Verify alignment
        if list(aligned_features.columns) != actual_training_columns:
            print("ERROR: Feature alignment failed!")
            print(f"Expected: {actual_training_columns[:5]}...")
            print(f"Got: {list(aligned_features.columns)[:5]}...")
            return None
        
        print("✓ Features successfully aligned with actual training data")
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

            # Handle AQI Category - Use exact same logic as training
            if 'aqi' in merged_data.columns:
                merged_data['aqi_category'] = merged_data['aqi'].apply(self.get_aqi_category)
            else:
                # If no AQI column, set default category
                merged_data['aqi_category'] = 'moderate'

            # Convert to string to ensure consistent type
            merged_data['aqi_category'] = merged_data['aqi_category'].astype(str)

            # One-hot encode - this might create different columns than training
            merged_data = pd.get_dummies(merged_data, columns=['aqi_category'], prefix='aqi_category')

            print(f"Features after initial processing: {len(merged_data.columns)} columns")

            # Get actual training feature columns (not from registry)
            actual_training_columns, actual_aqi_categories = self.load_actual_training_features()
            
            if actual_training_columns is None:
                print("Could not load actual training features")
                return None
            
            # Align features with actual training data
            aligned_features = self.align_features_with_actual_training(merged_data, actual_training_columns, actual_aqi_categories)
            
            if aligned_features is None:
                print("Feature alignment failed")
                return None

            # Return latest row for inference
            if len(aligned_features) == 0:
                print("Warning: No data available for inference")
                return None
                
            latest_data = aligned_features.iloc[-1:].copy()
            
            # Final validation
            print(f"Final feature matrix shape: {latest_data.shape}")
            print(f"Expected shape: (1, {len(actual_training_columns)})")
            
            return latest_data, actual_training_columns
            
        except Exception as e:
            print(f"Error in feature preparation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def make_predictions(self, features_df, actual_feature_columns):
        print("Making predictions for next 3 days...")
        try:
            if features_df is None or features_df.empty:
                print("No feature data available for prediction")
                return None
                
            model, scaler, model_info = self.model_registry.get_best_model(metric='test_overall_rmse')
            if model is None:
                print("No trained model found. Please run training pipeline first.")
                return None

            print(f"Using model: {model_info['model_name']}")
            
            # Get actual model feature count
            actual_model_features = self.get_actual_model_feature_count(model)
            print(f"Actual model expects: {actual_model_features} features")
            print(f"We have: {features_df.shape[1]} features")
            print(f"Registry says model expects: {len(model_info['feature_columns'].split(','))} features")

            # Prepare features for prediction
            X = features_df.select_dtypes(include=[np.number])
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Check for infinite values
            if np.isinf(X.values).any():
                print("Warning: Infinite values detected, replacing with 0")
                X = X.replace([np.inf, -np.inf], 0)

            # Verify feature count matches actual model
            if actual_model_features and X.shape[1] != actual_model_features:
                print(f"ERROR: Feature count mismatch with actual model!")
                print(f"Actual model expects: {actual_model_features}")
                print(f"We have: {X.shape[1]}")
                print(f"Feature difference: {X.shape[1] - actual_model_features}")
                return None

            # Apply scaling if model requires it
            if scaler is not None:
                print("Applying feature scaling...")
                X_scaled = scaler.transform(X.values)
                X_final = X_scaled
            else:
                X_final = X.values

            # Make prediction
            print("Generating predictions...")
            predictions = model.predict(X_final)
            
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(1, -1)
            
            prediction_values = predictions[0]  # Get first (and only) prediction
            print(f"Raw predictions: {prediction_values}")

            # Ensure we have 3 predictions
            if len(prediction_values) != 3:
                print(f"Warning: Expected 3 predictions, got {len(prediction_values)}")
                return None

            # Calculate 3-day average for compatibility
            avg_prediction = np.mean(prediction_values)

            # Get current AQI from features if available
            current_aqi = None
            if 'aqi' in features_df.columns:
                current_aqi = float(features_df['aqi'].iloc[0])

            prediction_result = {
                'timestamp': datetime.now(),
                'predicted_aqi_24h': float(prediction_values[0]),    # Day 1
                'predicted_aqi_48h': float(prediction_values[1]),    # Day 2  
                'predicted_aqi_72h': float(prediction_values[2]),    # Day 3
                'predicted_aqi_3day_avg': float(avg_prediction),     # Average for backward compatibility
                'model_used': model_info['model_name'],
                'model_timestamp': model_info['created_at'],
                'current_aqi': current_aqi
            }

            print(f"Day 1 (24h) prediction: {prediction_values[0]:.2f} AQI")
            print(f"Day 2 (48h) prediction: {prediction_values[1]:.2f} AQI")
            print(f"Day 3 (72h) prediction: {prediction_values[2]:.2f} AQI")
            print(f"3-day average: {avg_prediction:.2f} AQI")
            print(f"Model used: {model_info['model_name']}")
            
            return prediction_result

        except Exception as e:
            print(f"Error making predictions: {e}")
            import traceback
            traceback.print_exc()
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
                # Keep only last 100 predictions
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

        # Get individual day predictions
        day1_aqi = prediction_result['predicted_aqi_24h']
        day2_aqi = prediction_result['predicted_aqi_48h'] 
        day3_aqi = prediction_result['predicted_aqi_72h']
        avg_aqi = prediction_result['predicted_aqi_3day_avg']
        current_aqi = prediction_result.get('current_aqi', 0)

        alerts = []

        # Check each day for alerts
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

        # Check for significant changes from current AQI
        if current_aqi:
            for day_name, predicted_aqi in daily_predictions:
                if abs(predicted_aqi - current_aqi) > 50:
                    alerts.append({
                        'level': 'CHANGE_ALERT',
                        'day': day_name,
                        'message': f'{day_name} significant AQI change predicted: {current_aqi:.0f} → {predicted_aqi:.0f}',
                        'recommendation': 'Monitor air quality closely'
                    })

        # Check for trend across days
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
                all_alerts = all_alerts.tail(50)  # Keep only last 50 alerts
            else:
                all_alerts = alerts_df

            all_alerts.to_csv(alerts_file, index=False)

            print(f"Generated {len(alerts)} alerts")
            for alert in alerts:
                print(f"- {alert['level']} ({alert.get('day', 'N/A')}): {alert['message']}")

    def run_inference_pipeline(self):
        print("=== Starting Multi-Output Inference Pipeline ===")
        try:
            # Fetch latest data
            aqi_data, weather_data = self.fetch_latest_data()
            
            if aqi_data is None or aqi_data.empty:
                print("No AQI data available for inference")
                return None
                
            if weather_data is None or weather_data.empty:
                print("No weather data available for inference")
                return None
            
            # Prepare features using actual training data
            result = self.prepare_inference_features(aqi_data, weather_data)
            
            if result is None:
                print("Failed to prepare features")
                return None
                
            features_df, actual_feature_columns = result
            
            # Make predictions
            prediction_result = self.make_predictions(features_df, actual_feature_columns)
            
            if prediction_result is None:
                print("Failed to make predictions")
                return None
            
            # Save predictions and check alerts
            self.save_predictions(prediction_result)
            self.check_aqi_alerts(prediction_result)

            print("=== Multi-Output Inference Pipeline Completed Successfully ===")
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
    else:
        print("Inference pipeline failed!")