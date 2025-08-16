#backfill_pipeline.py

import sys
import os
from datetime import datetime

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from data_fetching.fetch_aqi_data import AQIDataFetcher
from feature_engineering.compute_features import FeatureEngineer
from feature_store.feature_store_manager import FeatureStore
from model_training.train_model import ModelTrainer

class BackfillPipeline:
    def __init__(self):
        self.data_fetcher = AQIDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.feature_store = FeatureStore()
        self.model_trainer = ModelTrainer()
        self.project_root = project_root  # Save root path

    def run_backfill_pipeline(self, skip_data_collection=True, start_date="2024-07-01"):
        """Run the complete backfill pipeline"""
        print("=== Starting Backfill Pipeline ===")
        print(f"Skip data collection: {skip_data_collection}")
        
        if not skip_data_collection:
            print(f"Start date: {start_date}")
            print(f"End date: {datetime.now().strftime('%Y-%m-%d')}")
        else:
            print("Using existing collected data")

        try:
            # Step 1: Fetch historical data (SKIP if data already collected)
            if skip_data_collection:
                print("1. Skipping data collection - using existing data...")
                # Verify that data exists
                if not self._verify_existing_data():
                    print("WARNING: No existing data found. You may need to enable data collection.")
                    print("Set skip_data_collection=False if you need to collect data.")
                    return False
                print("✓ Existing data verified")
            else:
                print("1. Fetching historical data...")
                self.data_fetcher.run_backfill(start_date)

            # Step 2: Compute features
            print("2. Computing features from existing data...")
            features_df = self.feature_engineer.compute_all_features()

            if features_df is None:
                print("Feature engineering failed. Stopping pipeline.")
                print("This might be because no raw data is available.")
                return False

            # Step 3: Save to feature store
            print("3. Saving to feature store...")
            feature_set_name = f"backfill_features_{datetime.now().strftime('%Y%m%d')}"
            self.feature_store.save_features_to_csv(features_df, feature_set_name)

            # Step 4: Train initial models
            print("4. Training initial multi-output models...")
            model_results = self.model_trainer.train_all_models()

            if model_results:
                print("=== Backfill Pipeline Completed Successfully ===")
                print(f"Features created: {len(features_df)} rows, {len(features_df.columns)} columns")
                print(f"Multi-output models trained: {len(model_results)}")

                for result in model_results:
                    print(f"- {result['model_name']}: Overall RMSE={result['test_overall_rmse']:.2f}")
                    print(f"  Day 1 RMSE={result['test_24h_rmse']:.2f}, Day 2 RMSE={result['test_48h_rmse']:.2f}, Day 3 RMSE={result['test_72h_rmse']:.2f}")

                return True
            else:
                print("Model training failed.")
                return False

        except Exception as e:
            print(f"Backfill pipeline failed: {e}")
            return False

    def _verify_existing_data(self):
        """Verify that existing data files are available"""
        try:
            # Check for raw data files
            raw_data_path = os.path.join(self.project_root, "data", "raw")
            
            if not os.path.exists(raw_data_path):
                print(f"Raw data directory not found: {raw_data_path}")
                return False
            
            # Look for data files (common patterns)
            data_files = []
            if os.path.exists(raw_data_path):
                for file in os.listdir(raw_data_path):
                    if file.endswith(('.csv', '.json', '.pkl')):
                        data_files.append(file)
            
            if len(data_files) == 0:
                print("No data files found in raw data directory")
                return False
            
            print(f"Found {len(data_files)} data files in {raw_data_path}")
            return True
            
        except Exception as e:
            print(f"Error verifying existing data: {e}")
            return False

    def validate_pipeline_setup(self):
        """Validate that all components are properly set up"""
        print("Validating pipeline setup...")

        required_dirs = ['data/raw', 'data/features', 'data/predictions', 'models']
        for dir_rel_path in required_dirs:
            dir_abs_path = os.path.join(self.project_root, dir_rel_path)
            if not os.path.exists(dir_abs_path):
                print(f"Creating missing directory: {dir_abs_path}")
                os.makedirs(dir_abs_path, exist_ok=True)

        print("Pipeline setup validation completed.")

    def cleanup_old_data(self, keep_days=30):
        """Clean up old data files"""
        print(f"Cleaning up data older than {keep_days} days...")

        # Clean up old feature files
        self.feature_store.cleanup_old_features(keep_last_n=5)

        # Clean up old model files
        from model_training.model_registry import ModelRegistry
        registry = ModelRegistry()
        registry.delete_old_models(keep_last_n=3)

        print("Cleanup completed.")

    def force_data_collection(self, start_date="2024-07-01"):
        """Force data collection even if skip_data_collection is True"""
        print("=== Forcing Data Collection ===")
        return self.run_backfill_pipeline(skip_data_collection=False, start_date=start_date)

if __name__ == "__main__":
    pipeline = BackfillPipeline()

    # Validate setup
    pipeline.validate_pipeline_setup()

    # Run backfill with data collection SKIPPED (since you already collected data)
    print("Running pipeline with existing collected data...")
    success = pipeline.run_backfill_pipeline(skip_data_collection=True)

    # If you need to force data collection again, uncomment the line below:
    # success = pipeline.force_data_collection()

    if success:
        print("Backfill pipeline completed successfully!")
    else:
        print("Backfill pipeline failed!")
        print("If you need to collect data, set skip_data_collection=False or use force_data_collection()")
