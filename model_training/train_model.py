

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
import os
from datetime import datetime
import sys

# Dynamic path resolution for project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add project root to sys.path for module imports
sys.path.append(project_root)

from feature_store.feature_store_manager import FeatureStore

class ModelTrainer:
    def __init__(self):
        self.feature_store = FeatureStore()
        self.models_path = os.path.join(project_root, "models")
        self.scaler = StandardScaler()

        # Ensure models directory exists
        os.makedirs(self.models_path, exist_ok=True)
        
    def load_training_data(self):
        features_df = self.feature_store.load_features_from_csv(latest=True)
        
        if features_df is None:
            print("No features found. Please run feature engineering first.")
            return None, None, None, None
            
        exclude_columns = ['timestamp', 'city', 'country', 'target_aqi_24h', 
                          'target_aqi_48h', 'target_aqi_72h', 'target_aqi_3day_avg']
        
        feature_columns = [col for col in features_df.columns if col not in exclude_columns]
        
        X = features_df[feature_columns].select_dtypes(include=[np.number])
        
        # Multi-output targets: predict AQI for next 3 individual days
        y = features_df[['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']]
        
        # Remove rows where any target is null
        valid_indices = ~y.isnull().any(axis=1)
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
        # Remove rows with too many missing features
        X = X.dropna(thresh=len(X.columns) * 0.7)
        y = y.loc[X.index]
        
        # Fill missing values in features
        X = X.fillna(X.mean())
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print(f"Feature columns: {list(X.columns)}")
        print(f"Target columns: {list(y.columns)}")
        
        return X, y, feature_columns, features_df

    def calculate_multioutput_metrics(self, y_true, y_pred, prefix=""):
        """Calculate metrics for multi-output regression"""
        metrics = {}
        target_names = ['24h', '48h', '72h']
        
        # Overall metrics
        overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        overall_mae = mean_absolute_error(y_true, y_pred)
        overall_r2 = r2_score(y_true, y_pred)
        
        metrics[f'{prefix}overall_rmse'] = overall_rmse
        metrics[f'{prefix}overall_mae'] = overall_mae
        metrics[f'{prefix}overall_r2'] = overall_r2
        
        # Individual target metrics
        for i, target_name in enumerate(target_names):
            rmse = np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y_true.iloc[:, i], y_pred[:, i])
            r2 = r2_score(y_true.iloc[:, i], y_pred[:, i])
            
            metrics[f'{prefix}{target_name}_rmse'] = rmse
            metrics[f'{prefix}{target_name}_mae'] = mae
            metrics[f'{prefix}{target_name}_r2'] = r2
        
        return metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        print("Training Random Forest for multi-output prediction...")
        
        rf_model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
        
        rf_model.fit(X_train, y_train)
        
        y_pred_train = rf_model.predict(X_train)
        y_pred_test = rf_model.predict(X_test)
        
        train_metrics = self.calculate_multioutput_metrics(y_train, y_pred_train, 'train_')
        test_metrics = self.calculate_multioutput_metrics(y_test, y_pred_test, 'test_')
        
        metrics = {
            'model_name': 'RandomForest_MultiOutput',
            **train_metrics,
            **test_metrics
        }
        
        return rf_model, metrics

    def train_ridge_regression(self, X_train, y_train, X_test, y_test):
        print("Training Ridge Regression for multi-output prediction...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        ridge_model = MultiOutputRegressor(
            Ridge(alpha=1.0, random_state=42)
        )
        
        ridge_model.fit(X_train_scaled, y_train)
        
        y_pred_train = ridge_model.predict(X_train_scaled)
        y_pred_test = ridge_model.predict(X_test_scaled)
        
        train_metrics = self.calculate_multioutput_metrics(y_train, y_pred_train, 'train_')
        test_metrics = self.calculate_multioutput_metrics(y_test, y_pred_test, 'test_')
        
        metrics = {
            'model_name': 'Ridge_MultiOutput',
            **train_metrics,
            **test_metrics
        }
        
        return ridge_model, metrics

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        print("Training XGBoost for multi-output prediction...")

        xgb_model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        )

        xgb_model.fit(X_train, y_train)

        y_pred_train = xgb_model.predict(X_train)
        y_pred_test = xgb_model.predict(X_test)

        train_metrics = self.calculate_multioutput_metrics(y_train, y_pred_train, 'train_')
        test_metrics = self.calculate_multioutput_metrics(y_test, y_pred_test, 'test_')

        metrics = {
            'model_name': 'XGBoost_MultiOutput',
            **train_metrics,
            **test_metrics
        }

        return xgb_model, metrics

    def save_model(self, model, model_name, metrics, feature_columns):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        model_filename = f"{model_name}_{timestamp}.joblib"
        model_path = os.path.join(self.models_path, model_filename)
        joblib.dump(model, model_path)
        
        scaler_filename = None
        if 'Ridge' in model_name:
            scaler_filename = f"scaler_{model_name}_{timestamp}.joblib"
            scaler_path = os.path.join(self.models_path, scaler_filename)
            joblib.dump(self.scaler, scaler_path)
        
        # Flatten metrics for CSV storage
        flattened_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)):
                flattened_metrics[key] = value
            else:
                flattened_metrics[key] = str(value)
        
        metadata = {
            'model_name': model_name,
            'model_file': model_filename,
            'scaler_file': scaler_filename,
            'created_at': timestamp,
            'feature_columns': ','.join(feature_columns),
            'num_features': len(feature_columns),
            **flattened_metrics
        }
        
        self._update_model_registry(metadata)
        print(f"Model saved: {model_path}")
        return model_path

    def _update_model_registry(self, metadata):
        registry_file = os.path.join(self.models_path, "model_registry.csv")
        metadata_df = pd.DataFrame([metadata])
        
        if os.path.exists(registry_file):
            existing_registry = pd.read_csv(registry_file)
            metadata_df = pd.concat([existing_registry, metadata_df], ignore_index=True)
            
        metadata_df.to_csv(registry_file, index=False)

    def train_all_models(self):
        print("Starting multi-output model training pipeline...")
        
        X, y, feature_columns, _ = self.load_training_data()
        
        if X is None:
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training set: X={X_train.shape}, y={y_train.shape}")
        print(f"Test set: X={X_test.shape}, y={y_test.shape}")
        
        models_results = []
        
        try:
            rf_model, rf_metrics = self.train_random_forest(X_train, y_train, X_test, y_test)
            self.save_model(rf_model, 'RandomForest_MultiOutput', rf_metrics, feature_columns)
            models_results.append(rf_metrics)
        except Exception as e:
            print(f"Error training Random Forest: {e}")
        
        try:
            ridge_model, ridge_metrics = self.train_ridge_regression(X_train, y_train, X_test, y_test)
            self.save_model(ridge_model, 'Ridge_MultiOutput', ridge_metrics, feature_columns)
            models_results.append(ridge_metrics)
        except Exception as e:
            print(f"Error training Ridge Regression: {e}")
        
        try:
            xgb_model, xgb_metrics = self.train_xgboost(X_train, y_train, X_test, y_test)
            self.save_model(xgb_model, 'XGBoost_MultiOutput', xgb_metrics, feature_columns)
            models_results.append(xgb_metrics)
        except Exception as e:
            print(f"Error training XGBoost: {e}")
        
        if models_results:
            print("\n=== Multi-Output Model Comparison ===")
            for result in models_results:
                print(f"\n{result['model_name']}:")
                print(f"  Overall - RMSE: {result['test_overall_rmse']:.2f}, MAE: {result['test_overall_mae']:.2f}, R²: {result['test_overall_r2']:.3f}")
                print(f"  Day 1 (24h) - RMSE: {result['test_24h_rmse']:.2f}, MAE: {result['test_24h_mae']:.2f}, R²: {result['test_24h_r2']:.3f}")
                print(f"  Day 2 (48h) - RMSE: {result['test_48h_rmse']:.2f}, MAE: {result['test_48h_mae']:.2f}, R²: {result['test_48h_r2']:.3f}")
                print(f"  Day 3 (72h) - RMSE: {result['test_72h_rmse']:.2f}, MAE: {result['test_72h_mae']:.2f}, R²: {result['test_72h_r2']:.3f}")
        
        return models_results

    def predict_next_3_days(self, model_path, input_features):
        """
        Make predictions for the next 3 days using a trained multi-output model
        
        Args:
            model_path: Path to the saved model
            input_features: DataFrame or array with current features
        
        Returns:
            dict: Predictions for each day
        """
        model = joblib.load(model_path)
        
        # Check if we need to scale features (for Ridge models)
        if 'Ridge' in model_path:
            scaler_path = model_path.replace('Ridge_MultiOutput', 'scaler_Ridge_MultiOutput')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                input_features = scaler.transform(input_features)
        
        predictions = model.predict(input_features)
        
        # Convert to readable format
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(1, -1)
        
        results = []
        for i in range(predictions.shape[0]):
            results.append({
                'day_1_aqi': predictions[i, 0],
                'day_2_aqi': predictions[i, 1],
                'day_3_aqi': predictions[i, 2]
            })
        
        return results

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.train_all_models()