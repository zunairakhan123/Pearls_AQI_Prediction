import pandas as pd
import os
import glob
from datetime import datetime
import numpy as np

class FeatureStore:
    def __init__(self):
        # Resolve paths dynamically based on current file location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        
        self.features_path = os.path.join(project_root, "data/features/")
        self.feature_metadata_file = os.path.join(self.features_path, "feature_metadata.csv")
        
        # Ensure features directory exists
        os.makedirs(self.features_path, exist_ok=True)
        
    def save_features_to_csv(self, features_df, feature_set_name=None):
        """Save features to CSV with metadata tracking"""
        if feature_set_name is None:
            feature_set_name = f"features_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
        filename = f"{feature_set_name}.csv"
        filepath = os.path.join(self.features_path, filename)
        features_df.to_csv(filepath, index=False)
        
        metadata = {
            'feature_set_name': feature_set_name,
            'filename': filename,
            'created_at': datetime.now(),
            'num_rows': len(features_df),
            'num_features': len(features_df.columns),
            'columns': ','.join(features_df.columns.tolist())
        }
        
        self._update_metadata(metadata)
        print(f"Features saved to {filepath}")
        return filepath
    
    def load_features_from_csv(self, feature_set_name=None, latest=True):
        """Load features from CSV"""
        if feature_set_name is None and latest:
            feature_files = glob.glob(os.path.join(self.features_path, "features_*.csv"))
            if not feature_files:
                print("No feature files found.")
                return None
                
            latest_file = max(feature_files, key=os.path.getmtime)
            print(f"Loading latest features from {latest_file}")
            return pd.read_csv(latest_file)
            
        elif feature_set_name:
            filepath = os.path.join(self.features_path, f"{feature_set_name}.csv")
            if os.path.exists(filepath):
                print(f"Loading features from {filepath}")
                return pd.read_csv(filepath)
            else:
                print(f"Feature set {feature_set_name} not found.")
                return None
                
        return None
    
    def list_feature_sets(self):
        """List all available feature sets"""
        if os.path.exists(self.feature_metadata_file):
            metadata_df = pd.read_csv(self.feature_metadata_file)
            return metadata_df
        else:
            print("No feature metadata found.")
            return pd.DataFrame()
    
    def _update_metadata(self, metadata):
        """Update feature metadata file"""
        metadata_df = pd.DataFrame([metadata])
        
        if os.path.exists(self.feature_metadata_file):
            existing_metadata = pd.read_csv(self.feature_metadata_file)
            metadata_df = pd.concat([existing_metadata, metadata_df], ignore_index=True)
            
        metadata_df.to_csv(self.feature_metadata_file, index=False)
    
    def get_feature_statistics(self, feature_set_name=None):
        """Get statistics for a feature set"""
        features_df = self.load_features_from_csv(feature_set_name)
        
        if features_df is not None:
            stats = {
                'shape': features_df.shape,
                'numeric_columns': features_df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': features_df.select_dtypes(include=['object']).columns.tolist(),
                'missing_values': features_df.isnull().sum().to_dict(),
                'summary_stats': features_df.describe().to_dict()
            }
            return stats
        return None
    
    def cleanup_old_features(self, keep_last_n=5):
        """Keep only the last N feature sets and remove older ones"""
        feature_files = glob.glob(os.path.join(self.features_path, "features_*.csv"))
        
        if len(feature_files) > keep_last_n:
            feature_files.sort(key=os.path.getmtime)
            files_to_remove = feature_files[:-keep_last_n]
            for file_path in files_to_remove:
                os.remove(file_path)
                print(f"Removed old feature file: {file_path}")

if __name__ == "__main__":
    fs = FeatureStore()
    feature_sets = fs.list_feature_sets()
    print("Available feature sets:")
    print(feature_sets)
    
    latest_features = fs.load_features_from_csv()
    if latest_features is not None:
        print(f"Loaded features with shape: {latest_features.shape}")
