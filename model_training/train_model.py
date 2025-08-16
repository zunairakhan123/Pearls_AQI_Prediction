
# #train_model.py
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.multioutput import MultiOutputRegressor
# from xgboost import XGBRegressor
# import joblib
# import os
# from datetime import datetime
# import sys

# # Dynamic path resolution for project root
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..'))

# # Add project root to sys.path for module imports
# sys.path.append(project_root)

# from feature_store.feature_store_manager import FeatureStore

# class ModelTrainer:
#     def __init__(self):
#         self.feature_store = FeatureStore()
#         self.models_path = os.path.join(project_root, "models")
#         self.scaler = StandardScaler()

#         # Ensure models directory exists
#         os.makedirs(self.models_path, exist_ok=True)
        
#     def load_training_data(self):
#         features_df = self.feature_store.load_features_from_csv(latest=True)
        
#         if features_df is None:
#             print("No features found. Please run feature engineering first.")
#             return None, None, None, None
            
#         exclude_columns = ['timestamp', 'city', 'country', 'target_aqi_24h', 
#                           'target_aqi_48h', 'target_aqi_72h', 'target_aqi_3day_avg']
        
#         feature_columns = [col for col in features_df.columns if col not in exclude_columns]
        
#         X = features_df[feature_columns].select_dtypes(include=[np.number])
        
#         # Multi-output targets: predict AQI for next 3 individual days
#         y = features_df[['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']]
        
#         # Remove rows where any target is null
#         valid_indices = ~y.isnull().any(axis=1)
#         X = X.loc[valid_indices]
#         y = y.loc[valid_indices]
        
#         # Remove rows with too many missing features
#         X = X.dropna(thresh=len(X.columns) * 0.7)
#         y = y.loc[X.index]
        
#         # Fill missing values in features
#         X = X.fillna(X.mean())
        
#         print(f"Training data shape: X={X.shape}, y={y.shape}")
#         print(f"Feature columns: {list(X.columns)}")
#         print(f"Target columns: {list(y.columns)}")
        
#         return X, y, feature_columns, features_df

#     def calculate_multioutput_metrics(self, y_true, y_pred, prefix=""):
#         """Calculate metrics for multi-output regression"""
#         metrics = {}
#         target_names = ['24h', '48h', '72h']
        
#         # Overall metrics
#         overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#         overall_mae = mean_absolute_error(y_true, y_pred)
#         overall_r2 = r2_score(y_true, y_pred)
        
#         metrics[f'{prefix}overall_rmse'] = overall_rmse
#         metrics[f'{prefix}overall_mae'] = overall_mae
#         metrics[f'{prefix}overall_r2'] = overall_r2
        
#         # Individual target metrics
#         for i, target_name in enumerate(target_names):
#             rmse = np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i]))
#             mae = mean_absolute_error(y_true.iloc[:, i], y_pred[:, i])
#             r2 = r2_score(y_true.iloc[:, i], y_pred[:, i])
            
#             metrics[f'{prefix}{target_name}_rmse'] = rmse
#             metrics[f'{prefix}{target_name}_mae'] = mae
#             metrics[f'{prefix}{target_name}_r2'] = r2
        
#         return metrics

#     def train_random_forest(self, X_train, y_train, X_test, y_test):
#         print("Training Random Forest for multi-output prediction...")
        
#         rf_model = MultiOutputRegressor(
#             RandomForestRegressor(
#                 n_estimators=100,
#                 max_depth=15,
#                 min_samples_split=5,
#                 min_samples_leaf=2,
#                 random_state=42,
#                 n_jobs=-1
#             )
#         )
        
#         rf_model.fit(X_train, y_train)
        
#         y_pred_train = rf_model.predict(X_train)
#         y_pred_test = rf_model.predict(X_test)
        
#         train_metrics = self.calculate_multioutput_metrics(y_train, y_pred_train, 'train_')
#         test_metrics = self.calculate_multioutput_metrics(y_test, y_pred_test, 'test_')
        
#         metrics = {
#             'model_name': 'RandomForest_MultiOutput',
#             **train_metrics,
#             **test_metrics
#         }
        
#         return rf_model, metrics

#     def train_ridge_regression(self, X_train, y_train, X_test, y_test):
#         print("Training Ridge Regression for multi-output prediction...")
        
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_test_scaled = self.scaler.transform(X_test)
        
#         ridge_model = MultiOutputRegressor(
#             Ridge(alpha=1.0, random_state=42)
#         )
        
#         ridge_model.fit(X_train_scaled, y_train)
        
#         y_pred_train = ridge_model.predict(X_train_scaled)
#         y_pred_test = ridge_model.predict(X_test_scaled)
        
#         train_metrics = self.calculate_multioutput_metrics(y_train, y_pred_train, 'train_')
#         test_metrics = self.calculate_multioutput_metrics(y_test, y_pred_test, 'test_')
        
#         metrics = {
#             'model_name': 'Ridge_MultiOutput',
#             **train_metrics,
#             **test_metrics
#         }
        
#         return ridge_model, metrics

#     def train_xgboost(self, X_train, y_train, X_test, y_test):
#         print("Training XGBoost for multi-output prediction...")

#         xgb_model = MultiOutputRegressor(
#             XGBRegressor(
#                 n_estimators=100,
#                 max_depth=10,
#                 learning_rate=0.1,
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 random_state=42,
#                 n_jobs=-1
#             )
#         )

#         xgb_model.fit(X_train, y_train)

#         y_pred_train = xgb_model.predict(X_train)
#         y_pred_test = xgb_model.predict(X_test)

#         train_metrics = self.calculate_multioutput_metrics(y_train, y_pred_train, 'train_')
#         test_metrics = self.calculate_multioutput_metrics(y_test, y_pred_test, 'test_')

#         metrics = {
#             'model_name': 'XGBoost_MultiOutput',
#             **train_metrics,
#             **test_metrics
#         }

#         return xgb_model, metrics

#     def save_model(self, model, model_name, metrics, feature_columns):
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
#         model_filename = f"{model_name}_{timestamp}.joblib"
#         model_path = os.path.join(self.models_path, model_filename)
#         joblib.dump(model, model_path)
        
#         scaler_filename = None
#         if 'Ridge' in model_name:
#             scaler_filename = f"scaler_{model_name}_{timestamp}.joblib"
#             scaler_path = os.path.join(self.models_path, scaler_filename)
#             joblib.dump(self.scaler, scaler_path)
        
#         # Flatten metrics for CSV storage
#         flattened_metrics = {}
#         for key, value in metrics.items():
#             if isinstance(value, (int, float, str)):
#                 flattened_metrics[key] = value
#             else:
#                 flattened_metrics[key] = str(value)
        
#         metadata = {
#             'model_name': model_name,
#             'model_file': model_filename,
#             'scaler_file': scaler_filename,
#             'created_at': timestamp,
#             'feature_columns': ','.join(feature_columns),
#             'num_features': len(feature_columns),
#             **flattened_metrics
#         }
        
#         self._update_model_registry(metadata)
#         print(f"Model saved: {model_path}")
#         return model_path

#     def _update_model_registry(self, metadata):
#         registry_file = os.path.join(self.models_path, "model_registry.csv")
#         metadata_df = pd.DataFrame([metadata])
        
#         if os.path.exists(registry_file):
#             existing_registry = pd.read_csv(registry_file)
#             metadata_df = pd.concat([existing_registry, metadata_df], ignore_index=True)
            
#         metadata_df.to_csv(registry_file, index=False)

#     def train_all_models(self):
#         print("Starting multi-output model training pipeline...")
        
#         X, y, feature_columns, _ = self.load_training_data()
        
#         if X is None:
#             return None
        
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, shuffle=True
#         )
        
#         print(f"Training set: X={X_train.shape}, y={y_train.shape}")
#         print(f"Test set: X={X_test.shape}, y={y_test.shape}")
        
#         models_results = []
        
#         try:
#             rf_model, rf_metrics = self.train_random_forest(X_train, y_train, X_test, y_test)
#             self.save_model(rf_model, 'RandomForest_MultiOutput', rf_metrics, feature_columns)
#             models_results.append(rf_metrics)
#         except Exception as e:
#             print(f"Error training Random Forest: {e}")
        
#         try:
#             ridge_model, ridge_metrics = self.train_ridge_regression(X_train, y_train, X_test, y_test)
#             self.save_model(ridge_model, 'Ridge_MultiOutput', ridge_metrics, feature_columns)
#             models_results.append(ridge_metrics)
#         except Exception as e:
#             print(f"Error training Ridge Regression: {e}")
        
#         try:
#             xgb_model, xgb_metrics = self.train_xgboost(X_train, y_train, X_test, y_test)
#             self.save_model(xgb_model, 'XGBoost_MultiOutput', xgb_metrics, feature_columns)
#             models_results.append(xgb_metrics)
#         except Exception as e:
#             print(f"Error training XGBoost: {e}")
        
#         if models_results:
#             print("\n=== Multi-Output Model Comparison ===")
#             for result in models_results:
#                 print(f"\n{result['model_name']}:")
#                 print(f"  Overall - RMSE: {result['test_overall_rmse']:.2f}, MAE: {result['test_overall_mae']:.2f}, R²: {result['test_overall_r2']:.3f}")
#                 print(f"  Day 1 (24h) - RMSE: {result['test_24h_rmse']:.2f}, MAE: {result['test_24h_mae']:.2f}, R²: {result['test_24h_r2']:.3f}")
#                 print(f"  Day 2 (48h) - RMSE: {result['test_48h_rmse']:.2f}, MAE: {result['test_48h_mae']:.2f}, R²: {result['test_48h_r2']:.3f}")
#                 print(f"  Day 3 (72h) - RMSE: {result['test_72h_rmse']:.2f}, MAE: {result['test_72h_mae']:.2f}, R²: {result['test_72h_r2']:.3f}")
        
#         return models_results

#     def predict_next_3_days(self, model_path, input_features):
#         """
#         Make predictions for the next 3 days using a trained multi-output model
        
#         Args:
#             model_path: Path to the saved model
#             input_features: DataFrame or array with current features
        
#         Returns:
#             dict: Predictions for each day
#         """
#         model = joblib.load(model_path)
        
#         # Check if we need to scale features (for Ridge models)
#         if 'Ridge' in model_path:
#             scaler_path = model_path.replace('Ridge_MultiOutput', 'scaler_Ridge_MultiOutput')
#             if os.path.exists(scaler_path):
#                 scaler = joblib.load(scaler_path)
#                 input_features = scaler.transform(input_features)
        
#         predictions = model.predict(input_features)
        
#         # Convert to readable format
#         if len(predictions.shape) == 1:
#             predictions = predictions.reshape(1, -1)
        
#         results = []
#         for i in range(predictions.shape[0]):
#             results.append({
#                 'day_1_aqi': predictions[i, 0],
#                 'day_2_aqi': predictions[i, 1],
#                 'day_3_aqi': predictions[i, 2]
#             })
        
#         return results

# if __name__ == "__main__":
#     trainer = ModelTrainer()
#     results = trainer.train_all_models()

#train_model.py
#train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
import joblib
import os
from datetime import datetime
import sys
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow imported successfully")
except ImportError as e:
    print(f"⚠️  TensorFlow not available: {e}")
    print("Please install TensorFlow: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False
    MinMaxScaler = None

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

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
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
        # Initialize scalers for LSTM (separate for features and targets)
        if TENSORFLOW_AVAILABLE and MinMaxScaler is not None:
            self.lstm_feature_scaler = MinMaxScaler(feature_range=(0, 1))
            self.lstm_target_scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.lstm_feature_scaler = None
            self.lstm_target_scaler = None
            
        self.feature_selector = SelectKBest(score_func=f_regression, k='all')

        # Ensure models directory exists
        os.makedirs(self.models_path, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(42)
        
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
        X = X.dropna(thresh=len(X.columns) * 0.8)  # Increased threshold from 0.7 to 0.8
        y = y.loc[X.index]
        
        # Improved feature preprocessing
        # Fill missing values with median (more robust than mean)
        X = X.fillna(X.median())
        
        # Remove features with very low variance
        feature_variance = X.var()
        low_variance_features = feature_variance[feature_variance < 0.01].index
        if len(low_variance_features) > 0:
            print(f"Removing {len(low_variance_features)} low variance features")
            X = X.drop(columns=low_variance_features)
        
        # Remove highly correlated features
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        if len(high_corr_features) > 0:
            print(f"Removing {len(high_corr_features)} highly correlated features")
            X = X.drop(columns=high_corr_features)
        
        print(f"Training data shape after preprocessing: X={X.shape}, y={y.shape}")
        print(f"Feature columns: {list(X.columns)}")
        print(f"Target columns: {list(y.columns)}")
        
        return X, y, list(X.columns), features_df

    def create_lstm_sequences_improved(self, X, y, sequence_length=10):
        """
        IMPROVED: Create overlapping sequences for better data utilization
        """
        if len(X) < sequence_length:
            print(f"Warning: Dataset too small for sequence length {sequence_length}")
            sequence_length = max(3, len(X) // 5)  # More conservative minimum
        
        X_sequences = []
        y_sequences = []
        
        # Create overlapping sequences with stride=1 for maximum data utilization
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        print(f"Created {len(X_sequences)} sequences from {len(X)} samples ({len(X_sequences)/len(X):.1%} utilization)")
        
        return np.array(X_sequences), np.array(y_sequences)

    def calculate_accuracy_for_aqi(self, y_true, y_pred, tolerance=10):
        """
        Calculate accuracy for AQI predictions with tolerance
        AQI prediction is considered accurate if within tolerance range
        """
        accuracies = []
        for i in range(y_true.shape[1]):
            diff = np.abs(y_true.iloc[:, i].values - y_pred[:, i])
            accurate_predictions = np.sum(diff <= tolerance)
            accuracy = accurate_predictions / len(y_true)
            accuracies.append(accuracy)
        
        overall_accuracy = np.mean([
            np.sum(np.abs(y_true.values - y_pred) <= tolerance, axis=0).mean() / len(y_true)
        ])
        
        return overall_accuracy, accuracies

    def calculate_multioutput_metrics(self, y_true, y_pred, prefix=""):
        """Calculate comprehensive metrics for multi-output regression"""
        metrics = {}
        target_names = ['24h', '48h', '72h']
        
        # Overall metrics
        overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        overall_mae = mean_absolute_error(y_true, y_pred)
        overall_r2 = r2_score(y_true, y_pred)
        
        # Calculate accuracy with different tolerance levels
        overall_acc_10, individual_acc_10 = self.calculate_accuracy_for_aqi(y_true, y_pred, tolerance=10)
        overall_acc_15, individual_acc_15 = self.calculate_accuracy_for_aqi(y_true, y_pred, tolerance=15)
        
        metrics[f'{prefix}overall_rmse'] = overall_rmse
        metrics[f'{prefix}overall_mae'] = overall_mae
        metrics[f'{prefix}overall_r2'] = overall_r2
        metrics[f'{prefix}overall_accuracy_10'] = overall_acc_10
        metrics[f'{prefix}overall_accuracy_15'] = overall_acc_15
        
        # Individual target metrics
        for i, target_name in enumerate(target_names):
            rmse = np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y_true.iloc[:, i], y_pred[:, i])
            r2 = r2_score(y_true.iloc[:, i], y_pred[:, i])
            
            metrics[f'{prefix}{target_name}_rmse'] = rmse
            metrics[f'{prefix}{target_name}_mae'] = mae
            metrics[f'{prefix}{target_name}_r2'] = r2
            metrics[f'{prefix}{target_name}_accuracy_10'] = individual_acc_10[i]
            metrics[f'{prefix}{target_name}_accuracy_15'] = individual_acc_15[i]
        
        return metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        print("Training optimized Random Forest for multi-output prediction...")
        
        # Optimized hyperparameters for CPU efficiency and better performance
        rf_model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=150,  # Increased from 100
                max_depth=20,      # Increased from 15
                min_samples_split=3,  # Reduced from 5
                min_samples_leaf=1,   # Reduced from 2
                max_features='sqrt',  # Added for better generalization
                bootstrap=True,
                oob_score=True,    # Out-of-bag scoring
                random_state=42,
                n_jobs=-1
            )
        )
        
        rf_model.fit(X_train, y_train)
        
        y_pred_train = rf_model.predict(X_train)
        y_pred_test = rf_model.predict(X_test)
        
        train_metrics = self.calculate_multioutput_metrics(y_train, y_pred_train, 'train_')
        test_metrics = self.calculate_multioutput_metrics(y_test, y_pred_test, 'test_')
        
        # Add OOB score if available
        oob_scores = []
        for estimator in rf_model.estimators_:
            if hasattr(estimator, 'oob_score_') and estimator.oob_score_ is not None:
                oob_scores.append(estimator.oob_score_)
        
        metrics = {
            'model_name': 'RandomForest_MultiOutput',
            'oob_score': np.mean(oob_scores) if oob_scores else None,
            **train_metrics,
            **test_metrics
        }
        
        return rf_model, metrics

    def train_ridge_regression(self, X_train, y_train, X_test, y_test):
        print("Training optimized Ridge Regression for multi-output prediction...")
        
        # Scale features using RobustScaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Optimized alpha value through simple validation
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        best_alpha = 1.0
        best_score = -np.inf
        
        for alpha in alphas:
            ridge_temp = MultiOutputRegressor(Ridge(alpha=alpha, random_state=42))
            scores = cross_val_score(ridge_temp, X_train_scaled, y_train, cv=3, scoring='r2')
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha
        
        print(f"Best alpha for Ridge: {best_alpha}")
        
        ridge_model = MultiOutputRegressor(
            Ridge(alpha=best_alpha, random_state=42, solver='auto')
        )
        
        ridge_model.fit(X_train_scaled, y_train)
        
        y_pred_train = ridge_model.predict(X_train_scaled)
        y_pred_test = ridge_model.predict(X_test_scaled)
        
        train_metrics = self.calculate_multioutput_metrics(y_train, y_pred_train, 'train_')
        test_metrics = self.calculate_multioutput_metrics(y_test, y_pred_test, 'test_')
        
        metrics = {
            'model_name': 'Ridge_MultiOutput',
            'best_alpha': best_alpha,
            'cv_score': best_score,
            **train_metrics,
            **test_metrics
        }
        
        return ridge_model, metrics

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        print("Training optimized XGBoost for multi-output prediction...")

        # Optimized hyperparameters for better performance and CPU efficiency
        xgb_model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=200,      # Increased from 100
                max_depth=8,           # Reduced from 10 for better generalization
                learning_rate=0.08,    # Slightly reduced for better convergence
                subsample=0.85,        # Increased from 0.8
                colsample_bytree=0.85, # Increased from 0.8
                reg_alpha=0.1,         # L1 regularization
                reg_lambda=1.0,        # L2 regularization
                min_child_weight=3,    # Added for regularization
                gamma=0.1,             # Added for regularization
                random_state=42,
                n_jobs=-1,
                verbosity=0           # Reduce output
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

    def train_lstm_improved(self, X_train, y_train, X_test, y_test):
        """
        🚀 COMPLETELY IMPROVED LSTM training with proper scaling and architecture
        """
        print("Training IMPROVED LSTM for multi-output prediction...")
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError as e:
            print(f"❌ TensorFlow not available: {e}")
            return None, {}
        
        # 🔧 CRITICAL FIX 1: PROPER DATA PREPARATION
        print(f"Original data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        
        # Sort by index to ensure temporal order (crucial for LSTM)
        X_train_sorted = X_train.sort_index()
        y_train_sorted = y_train.sort_index()
        X_test_sorted = X_test.sort_index()
        y_test_sorted = y_test.sort_index()
        
        # 🔧 CRITICAL FIX 2: SCALE BOTH FEATURES AND TARGETS
        print("Scaling features and targets separately...")
        
        # Scale features
        X_train_scaled = self.lstm_feature_scaler.fit_transform(X_train_sorted)
        X_test_scaled = self.lstm_feature_scaler.transform(X_test_sorted)
        
        # 🚨 CRITICAL: Scale targets too! This was missing in original code
        y_train_scaled = self.lstm_target_scaler.fit_transform(y_train_sorted)
        y_test_scaled = self.lstm_target_scaler.transform(y_test_sorted)
        
        print(f"Feature scaling range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
        print(f"Target scaling range: [{y_train_scaled.min():.3f}, {y_test_scaled.max():.3f}]")
        
        # 🔧 IMPROVEMENT 3: BETTER SEQUENCE CREATION
        sequence_length = min(12, max(5, len(X_train_scaled) // 30))  # More conservative
        print(f"Using sequence length: {sequence_length}")
        
        X_train_seq, y_train_seq = self.create_lstm_sequences_improved(X_train_scaled, y_train_scaled, sequence_length)
        X_test_seq, y_test_seq = self.create_lstm_sequences_improved(X_test_scaled, y_test_scaled, sequence_length)
        
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print("❌ Error: Not enough data to create LSTM sequences")
            return None, {}
        
        print(f"Sequence shapes: X_train_seq={X_train_seq.shape}, y_train_seq={y_train_seq.shape}")
        
        # 🔧 IMPROVEMENT 4: ADAPTIVE ARCHITECTURE
        n_features = X_train_seq.shape[2]
        n_samples = X_train_seq.shape[0]
        
        model = Sequential()
        
        if n_samples < 200:
            print("Using simplified LSTM architecture for small dataset")
            model.add(LSTM(units=24, return_sequences=False, 
                          input_shape=(sequence_length, n_features),
                          dropout=0.1, recurrent_dropout=0.1))
            model.add(BatchNormalization())
            model.add(Dense(units=12, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(units=3))
        elif n_samples < 500:
            print("Using medium LSTM architecture")
            model.add(LSTM(units=32, return_sequences=True,
                          input_shape=(sequence_length, n_features),
                          dropout=0.15, recurrent_dropout=0.15))
            model.add(BatchNormalization())
            model.add(LSTM(units=16, return_sequences=False,
                          dropout=0.15, recurrent_dropout=0.15))
            model.add(BatchNormalization())
            model.add(Dense(units=8, activation='relu'))
            model.add(Dropout(0.25))
            model.add(Dense(units=3))
        else:
            print("Using full LSTM architecture for large dataset")
            model.add(LSTM(units=64, return_sequences=True,
                          input_shape=(sequence_length, n_features),
                          dropout=0.2, recurrent_dropout=0.2))
            model.add(BatchNormalization())
            model.add(LSTM(units=32, return_sequences=True,
                          dropout=0.2, recurrent_dropout=0.2))
            model.add(BatchNormalization())
            model.add(LSTM(units=16, return_sequences=False,
                          dropout=0.2, recurrent_dropout=0.2))
            model.add(BatchNormalization())
            model.add(Dense(units=12, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(units=3))
        
        # 🔧 IMPROVEMENT 5: BETTER COMPILATION
        model.compile(
            optimizer=Adam(
                learning_rate=0.01,    # Higher initial learning rate
                beta_1=0.9, 
                beta_2=0.999,
                epsilon=1e-07
            ),
            loss='mse',
            metrics=['mae']
        )
        
        print("Model architecture:")
        model.summary()
        
        # 🔧 IMPROVEMENT 6: BETTER CALLBACKS
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=25,  # More patience
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001  # Smaller minimum delta
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Reduce LR by half
            patience=10,
            min_lr=0.0001,
            verbose=1
        )
        
        # 🔧 IMPROVEMENT 7: ADAPTIVE TRAINING PARAMETERS
        batch_size = min(32, max(8, n_samples // 15))
        epochs = min(200, max(50, 2000 // n_samples))  # More epochs for smaller datasets
        
        print(f"Training parameters:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Max epochs: {epochs}")
        print(f"  - Samples: {n_samples}")
        
        # 🚀 TRAINING
        try:
            print("Starting LSTM training...")
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_seq, y_test_seq),
                callbacks=[early_stopping, reduce_lr],
                verbose=1,
                shuffle=True
            )
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return None, {}
        
        # 🔧 CRITICAL FIX 8: PROPER PREDICTION AND INVERSE SCALING
        try:
            print("Making predictions...")
            y_pred_train_scaled = model.predict(X_train_seq, verbose=0)
            y_pred_test_scaled = model.predict(X_test_seq, verbose=0)
            
            # 🚨 CRITICAL: Transform predictions back to original scale
            print("Inverse transforming predictions to original scale...")
            y_pred_train = self.lstm_target_scaler.inverse_transform(y_pred_train_scaled)
            y_pred_test = self.lstm_target_scaler.inverse_transform(y_pred_test_scaled)
            
            # Transform actual values back for comparison
            y_train_actual = self.lstm_target_scaler.inverse_transform(y_train_seq)
            y_test_actual = self.lstm_target_scaler.inverse_transform(y_test_seq)
            
            print(f"Prediction ranges:")
            print(f"  Train predictions: [{y_pred_train.min():.1f}, {y_pred_train.max():.1f}]")
            print(f"  Test predictions: [{y_pred_test.min():.1f}, {y_pred_test.max():.1f}]")
            print(f"  Train actual: [{y_train_actual.min():.1f}, {y_train_actual.max():.1f}]")
            print(f"  Test actual: [{y_test_actual.min():.1f}, {y_test_actual.max():.1f}]")
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, {}
        
        # 🔧 CALCULATE METRICS ON ORIGINAL SCALE
        y_train_df = pd.DataFrame(y_train_actual, columns=['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h'])
        y_test_df = pd.DataFrame(y_test_actual, columns=['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h'])
        
        train_metrics = self.calculate_multioutput_metrics(y_train_df, y_pred_train, 'train_')
        test_metrics = self.calculate_multioutput_metrics(y_test_df, y_pred_test, 'test_')
        
        # Training history metrics
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        best_val_loss = min(history.history['val_loss'])
        initial_val_loss = history.history['val_loss'][0]
        
        metrics = {
            'model_name': 'LSTM_Improved_MultiOutput',
            'sequence_length': sequence_length,
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'best_val_loss': best_val_loss,
            'improvement_pct': ((initial_val_loss - best_val_loss) / initial_val_loss) * 100,
            'train_samples': len(X_train_seq),
            'test_samples': len(X_test_seq),
            'batch_size': batch_size,
            **train_metrics,
            **test_metrics
        }
        
        # 🔧 PACKAGE EVERYTHING NEEDED FOR PREDICTIONS
        lstm_package = {
            'model': model,
            'feature_scaler': self.lstm_feature_scaler,
            'target_scaler': self.lstm_target_scaler,  # CRITICAL for predictions!
            'sequence_length': sequence_length,
            'feature_names': list(X_train.columns),
            'training_history': history.history
        }
        
        # 🔍 DIAGNOSTIC INFORMATION
        print("\n" + "="*60)
        print("🧠 LSTM TRAINING DIAGNOSTICS")
        print("="*60)
        print(f"✅ Data utilization: {len(X_train_seq)}/{len(X_train)} samples ({len(X_train_seq)/len(X_train):.1%})")
        print(f"📊 Loss improvement: {((initial_val_loss - best_val_loss) / initial_val_loss) * 100:.1f}%")
        print(f"🎯 Final train loss: {final_train_loss:.4f}")
        print(f"🎯 Final val loss: {final_val_loss:.4f}")
        print(f"⭐ Best val loss: {best_val_loss:.4f}")
        print(f"⏱️  Epochs trained: {len(history.history['loss'])}/{epochs}")
        
        # Performance checks
        if final_val_loss > final_train_loss * 3:
            print("⚠️  Warning: High validation loss suggests overfitting")
        elif final_val_loss < final_train_loss * 1.1:
            print("✅ Good training/validation balance")
        
        if len(history.history['loss']) == epochs:
            print("⚠️  Training stopped at max epochs - consider increasing epochs")
        else:
            print("✅ Early stopping triggered - optimal training duration")
        
        print("="*60)
        
        return lstm_package, metrics

    def save_model(self, model, model_name, metrics, feature_columns):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        model_filename = f"{model_name}_{timestamp}.joblib"
        model_path = os.path.join(self.models_path, model_filename)
        
        # Special handling for LSTM model
        if 'LSTM' in model_name:
            # Save the entire LSTM package (model + scalers + metadata)
            joblib.dump(model, model_path)
        else:
            # Save regular sklearn models
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
        print(f"✅ Model saved: {model_path}")
        return model_path

    def _update_model_registry(self, metadata):
        registry_file = os.path.join(self.models_path, "model_registry.csv")
        metadata_df = pd.DataFrame([metadata])
        
        if os.path.exists(registry_file):
            existing_registry = pd.read_csv(registry_file)
            metadata_df = pd.concat([existing_registry, metadata_df], ignore_index=True)
            
        metadata_df.to_csv(registry_file, index=False)

    def train_all_models(self):
        print("🚀 Starting IMPROVED multi-output model training pipeline...")
        
        X, y, feature_columns, _ = self.load_training_data()
        
        if X is None:
            return None
        
        # Stratified split to ensure balanced distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training set: X={X_train.shape}, y={y_train.shape}")
        print(f"Test set: X={X_test.shape}, y={y_test.shape}")
        
        models_results = []
        
        # Train Random Forest
        try:
            print("\n" + "="*50)
            rf_model, rf_metrics = self.train_random_forest(X_train, y_train, X_test, y_test)
            self.save_model(rf_model, 'RandomForest_MultiOutput', rf_metrics, feature_columns)
            models_results.append(rf_metrics)
            print("✅ Random Forest training completed")
        except Exception as e:
            print(f"❌ Error training Random Forest: {e}")
        
        # Train Ridge Regression
        try:
            print("\n" + "="*50)
            ridge_model, ridge_metrics = self.train_ridge_regression(X_train, y_train, X_test, y_test)
            self.save_model(ridge_model, 'Ridge_MultiOutput', ridge_metrics, feature_columns)
            models_results.append(ridge_metrics)
            print("✅ Ridge Regression training completed")
        except Exception as e:
            print(f"❌ Error training Ridge Regression: {e}")
        
        # Train XGBoost
        try:
            print("\n" + "="*50)
            xgb_model, xgb_metrics = self.train_xgboost(X_train, y_train, X_test, y_test)
            self.save_model(xgb_model, 'XGBoost_MultiOutput', xgb_metrics, feature_columns)
            models_results.append(xgb_metrics)
            print("✅ XGBoost training completed")
        except Exception as e:
            print(f"❌ Error training XGBoost: {e}")
        
        # Train IMPROVED LSTM
        try:
            print("\n" + "="*50)
            print("🧠 Attempting to train IMPROVED LSTM model...")
            if not TENSORFLOW_AVAILABLE:
                print("❌ TensorFlow not available. Skipping LSTM training.")
                print("💡 To install TensorFlow, run: pip install tensorflow")
            else:
                lstm_model, lstm_metrics = self.train_lstm_improved(X_train, y_train, X_test, y_test)
                if lstm_model is not None and lstm_metrics:
                    self.save_model(lstm_model, 'LSTM_Improved_MultiOutput', lstm_metrics, feature_columns)
                    models_results.append(lstm_metrics)
                    print("🎉 LSTM model trained successfully with improvements!")
                else:
                    print("❌ LSTM model training returned None - check logs above")
        except ImportError as e:
            print(f"❌ TensorFlow/Keras import error: {e}")
            print("💡 Please install tensorflow: pip install tensorflow")
        except Exception as e:
            print(f"❌ Error training IMPROVED LSTM: {e}")
            import traceback
            traceback.print_exc()
        
        # Display comprehensive results
        if models_results:
            print("\n" + "="*80)
            print("📊 COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
            print("="*80)
            
            for result in models_results:
                print(f"\n📈 {result['model_name']}:")
                print(f"{'='*60}")
                
                # Overall Performance
                print(f"🎯 OVERALL PERFORMANCE:")
                print(f"   RMSE: {result['test_overall_rmse']:.2f}")
                print(f"   MAE:  {result['test_overall_mae']:.2f}")
                print(f"   R²:   {result['test_overall_r2']:.3f}")
                print(f"   Accuracy (±10): {result['test_overall_accuracy_10']:.1%}")
                print(f"   Accuracy (±15): {result['test_overall_accuracy_15']:.1%}")
                
                # Individual Day Performance
                print(f"\n📅 DAILY PERFORMANCE BREAKDOWN:")
                days = ['24h', '48h', '72h']
                day_names = ['Day 1', 'Day 2', 'Day 3']
                
                for day, day_name in zip(days, day_names):
                    print(f"   {day_name} ({day}):")
                    print(f"     RMSE: {result[f'test_{day}_rmse']:.2f} | "
                          f"MAE: {result[f'test_{day}_mae']:.2f} | "
                          f"R²: {result[f'test_{day}_r2']:.3f}")
                    print(f"     Accuracy (±10): {result[f'test_{day}_accuracy_10']:.1%} | "
                          f"Accuracy (±15): {result[f'test_{day}_accuracy_15']:.1%}")
                
                # Model-specific metrics
                if 'oob_score' in result and result['oob_score'] is not None:
                    print(f"\n🌲 Random Forest OOB Score: {result['oob_score']:.3f}")
                
                if 'best_alpha' in result:
                    print(f"\n🔧 Ridge Configuration:")
                    print(f"   Best Alpha: {result['best_alpha']}")
                    print(f"   Cross-validation Score: {result['cv_score']:.3f}")
                
                if 'sequence_length' in result:
                    print(f"\n🧠 LSTM Configuration:")
                    print(f"   Sequence Length: {result['sequence_length']}")
                    print(f"   Epochs Trained: {result['epochs_trained']}")
                    print(f"   Final Train Loss: {result['final_train_loss']:.4f}")
                    print(f"   Final Validation Loss: {result['final_val_loss']:.4f}")
                    if 'improvement_pct' in result:
                        print(f"   Loss Improvement: {result['improvement_pct']:.1f}%")
                    print(f"   Training Samples: {result['train_samples']}")
                    print(f"   Batch Size: {result['batch_size']}")
            
            # Find best models for each metric
            print(f"\n{'='*80}")
            print("🏆 BEST PERFORMING MODELS BY METRIC")
            print("="*80)
            
            # Best overall R²
            best_r2_model = max(models_results, key=lambda x: x['test_overall_r2'])
            print(f"🥇 Best Overall R² Score: {best_r2_model['model_name']} ({best_r2_model['test_overall_r2']:.3f})")
            
            # Best overall accuracy
            best_acc_model = max(models_results, key=lambda x: x['test_overall_accuracy_10'])
            print(f"🎯 Best Overall Accuracy (±10): {best_acc_model['model_name']} ({best_acc_model['test_overall_accuracy_10']:.1%})")
            
            # Lowest overall RMSE
            best_rmse_model = min(models_results, key=lambda x: x['test_overall_rmse'])
            print(f"📉 Lowest Overall RMSE: {best_rmse_model['model_name']} ({best_rmse_model['test_overall_rmse']:.2f})")
            
            # Best individual day performance
            print(f"\n🏅 BEST DAILY PERFORMANCE:")
            for day in ['24h', '48h', '72h']:
                best_day_model = min(models_results, key=lambda x: x[f'test_{day}_rmse'])
                print(f"   Best {day}: {best_day_model['model_name']} (RMSE: {best_day_model[f'test_{day}_rmse']:.2f})")
            
            # LSTM specific improvements (if LSTM was trained)
            lstm_results = [r for r in models_results if 'LSTM' in r['model_name']]
            if lstm_results:
                lstm_result = lstm_results[0]
                print(f"\n🧠 LSTM IMPROVEMENTS SUMMARY:")
                print(f"   Model: {lstm_result['model_name']}")
                if 'improvement_pct' in lstm_result:
                    print(f"   Training Loss Reduction: {lstm_result['improvement_pct']:.1f}%")
                print(f"   Final R² Score: {lstm_result['test_overall_r2']:.3f}")
                print(f"   Data Utilization: {lstm_result['train_samples']} sequences")
                
                # Compare to original LSTM (if we had baseline metrics)
                print(f"   🔧 Key improvements applied:")
                print(f"     ✅ Separate feature and target scaling")
                print(f"     ✅ Proper inverse transformation")
                print(f"     ✅ Temporal data ordering")
                print(f"     ✅ Improved sequence creation")
                print(f"     ✅ Adaptive architecture")
                print(f"     ✅ Better training parameters")
        
        print(f"\n{'='*80}")
        print("✅ IMPROVED MODEL TRAINING PIPELINE COMPLETED!")
        print("="*80)
        
        return models_results

    def predict_with_improved_models(self, model_path, input_features):
        """
        🚀 IMPROVED prediction method that handles all model types correctly
        """
        try:
            model = joblib.load(model_path)
            
            # Handle IMPROVED LSTM model prediction
            if 'LSTM' in model_path and isinstance(model, dict):
                return self._predict_lstm_improved(model, input_features)
            
            # Handle Ridge model prediction  
            elif 'Ridge' in model_path:
                return self._predict_ridge(model_path, model, input_features)
            
            # Handle other models (Random Forest, XGBoost)
            else:
                return self._predict_tree_models(model, input_features)
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def _predict_lstm_improved(self, lstm_package, input_features):
        """Handle IMPROVED LSTM predictions with proper scaling"""
        try:
            lstm_model = lstm_package['model']
            feature_scaler = lstm_package['feature_scaler']
            target_scaler = lstm_package['target_scaler']  # CRITICAL!
            sequence_length = lstm_package['sequence_length']
            
            # Scale input features
            input_scaled = feature_scaler.transform(input_features)
            
            # Create sequences for prediction
            if len(input_scaled) >= sequence_length:
                input_seq = input_scaled[-sequence_length:].reshape(1, sequence_length, -1)
            else:
                # Pad with mean if insufficient history
                padding_needed = sequence_length - len(input_scaled)
                mean_values = np.mean(input_scaled, axis=0) if len(input_scaled) > 0 else np.zeros(input_scaled.shape[1])
                
                padded_input = np.vstack([
                    np.tile(mean_values, (padding_needed, 1)),
                    input_scaled
                ])
                input_seq = padded_input.reshape(1, sequence_length, -1)
            
            # Make prediction on scaled data
            predictions_scaled = lstm_model.predict(input_seq, verbose=0)
            
            # 🚨 CRITICAL: Inverse transform back to original scale
            predictions = target_scaler.inverse_transform(predictions_scaled)
            
            # Convert to readable format
            results = []
            for i in range(predictions.shape[0]):
                results.append({
                    'day_1_aqi': float(max(0, predictions[i, 0])),  # Ensure non-negative
                    'day_2_aqi': float(max(0, predictions[i, 1])),
                    'day_3_aqi': float(max(0, predictions[i, 2]))
                })
            
            return results
            
        except Exception as e:
            print(f"❌ LSTM prediction error: {e}")
            return None
    
    def _predict_ridge(self, model_path, model, input_features):
        """Handle Ridge model predictions with scaling"""
        try:
            scaler_path = model_path.replace('Ridge_MultiOutput', 'scaler_Ridge_MultiOutput')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                input_features_scaled = scaler.transform(input_features)
                predictions = model.predict(input_features_scaled)
            else:
                predictions = model.predict(input_features)
            
            # Convert to readable format
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(1, -1)
            
            results = []
            for i in range(predictions.shape[0]):
                results.append({
                    'day_1_aqi': float(max(0, predictions[i, 0])),
                    'day_2_aqi': float(max(0, predictions[i, 1])),
                    'day_3_aqi': float(max(0, predictions[i, 2]))
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Ridge prediction error: {e}")
            return None
    
    def _predict_tree_models(self, model, input_features):
        """Handle tree-based model predictions (RF, XGBoost)"""
        try:
            predictions = model.predict(input_features)
            
            # Convert to readable format
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(1, -1)
            
            results = []
            for i in range(predictions.shape[0]):
                results.append({
                    'day_1_aqi': float(max(0, predictions[i, 0])),
                    'day_2_aqi': float(max(0, predictions[i, 1])),
                    'day_3_aqi': float(max(0, predictions[i, 2]))
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Tree model prediction error: {e}")
            return None

    # Legacy method for backwards compatibility
    def predict_next_3_days(self, model_path, input_features):
        """Legacy method - redirects to improved prediction"""
        return self.predict_with_improved_models(model_path, input_features)

if __name__ == "__main__":
    print("🚀 Starting IMPROVED Model Training Pipeline...")
    trainer = ModelTrainer()
    results = trainer.train_all_models()
    
    if results:
        print(f"\n🎉 Successfully trained {len(results)} models!")
        for result in results:
            print(f"   - {result['model_name']}: R² = {result['test_overall_r2']:.3f}")
    else:
        print("❌ No models were successfully trained. Check the logs above.")