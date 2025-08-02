import pandas as pd
import joblib
import os
from datetime import datetime

class ModelRegistry:
    def __init__(self):
        # Dynamically resolve the models directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        
        self.models_path = os.path.join(project_root, "models")
        self.registry_file = os.path.join(self.models_path, "model_registry.csv")
        
        # Ensure models directory exists
        os.makedirs(self.models_path, exist_ok=True)
        
    def get_latest_model(self, model_name=None):
        """Get the latest trained model"""
        if not os.path.exists(self.registry_file):
            print("No models found in registry.")
            return None, None, None
            
        registry_df = pd.read_csv(self.registry_file)
        
        if model_name:
            registry_df = registry_df[registry_df['model_name'] == model_name]
            
        if registry_df.empty:
            print(f"No models found for {model_name}")
            return None, None, None
            
        latest_model = registry_df.loc[registry_df['created_at'].idxmax()]
        
        # Load model
        model_path = os.path.join(self.models_path, latest_model['model_file'])
        model = joblib.load(model_path)
        
        # Load scaler if exists
        scaler = None
        if pd.notna(latest_model['scaler_file']):
            scaler_path = os.path.join(self.models_path, latest_model['scaler_file'])
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
        
        print(f"Loaded model: {latest_model['model_name']} from {latest_model['created_at']}")
        return model, scaler, latest_model
    
    def get_best_model(self, metric='test_rmse', ascending=True):
        """Get the best model based on specified metric"""
        if not os.path.exists(self.registry_file):
            print("No models found in registry.")
            return None, None, None
            
        registry_df = pd.read_csv(self.registry_file)
        
        if metric not in registry_df.columns:
            print(f"Metric {metric} not found in registry.")
            return None, None, None
            
        best_model = registry_df.loc[registry_df[metric].idxmin() if ascending else registry_df[metric].idxmax()]
        
        # Load model
        model_path = os.path.join(self.models_path, best_model['model_file'])
        model = joblib.load(model_path)
        
        # Load scaler if exists
        scaler = None
        if pd.notna(best_model['scaler_file']):
            scaler_path = os.path.join(self.models_path, best_model['scaler_file'])
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
        
        print(f"Loaded best model: {best_model['model_name']} with {metric}={best_model[metric]:.3f}")
        return model, scaler, best_model
    
    def list_all_models(self):
        """List all models in the registry"""
        if not os.path.exists(self.registry_file):
            print("No models found in registry.")
            return pd.DataFrame()
            
        registry_df = pd.read_csv(self.registry_file)
        return registry_df
    
    def delete_old_models(self, keep_last_n=3):
        """Delete old model files, keeping only the last N models"""
        if not os.path.exists(self.registry_file):
            return
            
        registry_df = pd.read_csv(self.registry_file)
        registry_df = registry_df.sort_values('created_at')
        
        if len(registry_df) > keep_last_n:
            models_to_delete = registry_df.iloc[:-keep_last_n]
            
            for _, model_info in models_to_delete.iterrows():
                model_path = os.path.join(self.models_path, model_info['model_file'])
                if os.path.exists(model_path):
                    os.remove(model_path)
                    print(f"Deleted old model: {model_path}")
                
                if pd.notna(model_info['scaler_file']):
                    scaler_path = os.path.join(self.models_path, model_info['scaler_file'])
                    if os.path.exists(scaler_path):
                        os.remove(scaler_path)
                        print(f"Deleted old scaler: {scaler_path}")
            
            updated_registry = registry_df.iloc[-keep_last_n:]
            updated_registry.to_csv(self.registry_file, index=False)
            print(f"Updated registry, kept last {keep_last_n} models")

if __name__ == "__main__":
    registry = ModelRegistry()
    
    # List all models
    models = registry.list_all_models()
    print("All models in registry:")
    print(models)
    
    # Get best model
    best_model, scaler, model_info = registry.get_best_model()
    if best_model:
        print(f"Best model loaded: {model_info['model_name']}")
