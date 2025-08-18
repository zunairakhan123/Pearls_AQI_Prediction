import os
import sys
import pandas as pd
from huggingface_hub import HfApi

def upload_best_model_to_hub():
    """
    Reads the model registry, finds the best model, and uploads its files to the Hugging Face Hub.
    """
    
    # 1. Get environment variables
    HF_REPO = os.environ.get('HF_REPO')
    HF_TOKEN = os.environ.get('HF_TOKEN')

    if not HF_REPO or not HF_TOKEN:
        print("❌ Error: Hugging Face repository or token not found. Please set HF_REPO and HF_TOKEN environment variables.")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)
    
    # 2. Find the model registry file
    registry_file_path = os.path.join(os.getcwd(), 'models', 'model_registry.csv')
    
    if not os.path.exists(registry_file_path):
        print(f"❌ Error: Model registry not found at {registry_file_path}. Cannot determine best model to upload.")
        sys.exit(1)

    # 3. Read the registry and find the best model
    try:
        registry_df = pd.read_csv(registry_file_path)
        
        # We assume the best model has the lowest overall test RMSE
        best_model_row = registry_df.loc[registry_df['test_overall_rmse'].idxmin()]
        
        best_model_name = best_model_row['model_name']
        best_model_file = best_model_row['model_file']
        
        print(f"🎯 Best Model Found: {best_model_name} with RMSE: {best_model_row['test_overall_rmse']:.2f}")
        
    except KeyError:
        print("❌ Error: Missing required columns in model_registry.csv (e.g., 'test_overall_rmse').")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading or processing model registry: {e}")
        sys.exit(1)
        
    model_path = os.path.join(os.getcwd(), 'models', best_model_file)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Best model file not found at {model_path}. Please check your model training script.")
        sys.exit(1)

    # 4. Upload the model and associated files
    try:
        print(f"✅ Model meets criteria, uploading '{best_model_file}' to Hugging Face Hub...")
        
        # Upload the model file
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=f'models/{best_model_file}',
            repo_id=HF_REPO,
            repo_type="model",
        )
        print("✅ Model upload successful!")
        
        # Upload the scaler file if it exists
        scaler_file = best_model_row.get('scaler_file')
        if scaler_file and pd.notna(scaler_file):
            scaler_path = os.path.join(os.getcwd(), 'models', scaler_file)
            if os.path.exists(scaler_path):
                print(f"✅ Uploading scaler file '{scaler_file}'...")
                api.upload_file(
                    path_or_fileobj=scaler_path,
                    path_in_repo=f'models/{scaler_file}',
                    repo_id=HF_REPO,
                    repo_type="model",
                )
                print("✅ Scaler upload successful!")
                
    except Exception as e:
        print(f"❌ Failed to upload files to Hugging Face Hub: {e}")
        sys.exit(1)

if __name__ == "__main__":
    upload_best_model_to_hub()