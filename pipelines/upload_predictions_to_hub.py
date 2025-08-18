# pipelines/upload_predictions_to_hub.py
import os
import glob
from datetime import datetime
from huggingface_hub import HfApi, login

def main():
    """
    Uploads prediction and alert files to a specified Hugging Face Hub repository.
    """
    # Get environment variables set in the GitHub Actions workflow
    hf_token = os.environ.get('HF_TOKEN')
    repo_id = os.environ.get('HF_REPO')

    if not hf_token or not repo_id:
        print("❌ Error: HF_TOKEN or HF_REPO environment variables are not set.")
        exit(1)

    try:
        # Log in to Hugging Face Hub
        login(token=hf_token)
        api = HfApi()

        print("✅ Logged in to Hugging Face Hub successfully.")

        # Define the base paths for the files to upload
        prediction_path = 'predictions/latest_prediction.csv'
        alerts_path = 'predictions/aqi_alerts.csv'

        # Upload latest prediction and alerts files
        print("Starting upload of latest predictions and alerts...")
        if os.path.exists(prediction_path):
            api.upload_file(
                path_or_fileobj=prediction_path,
                path_in_repo='predictions/latest_prediction.csv',
                repo_id=repo_id,
                commit_message='🔮 Upload latest predictions'
            )
            print(f'✅ Uploaded {prediction_path} to Hugging Face Hub')
        else:
            print(f'⚠️ Warning: {prediction_path} not found. Skipping upload.')

        if os.path.exists(alerts_path):
            api.upload_file(
                path_or_fileobj=alerts_path,
                path_in_repo='predictions/aqi_alerts.csv',
                repo_id=repo_id,
                commit_message='🚨 Upload AQI alerts'
            )
            print(f'✅ Uploaded {alerts_path} to Hugging Face Hub')
        else:
            print(f'⚠️ Warning: {alerts_path} not found. Skipping upload.')

        # Upload timestamped predictions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prediction_files = glob.glob('predictions/*.csv') + glob.glob('predictions/*.json')
        
        print("Checking for timestamped prediction files to upload...")
        for pred_file in prediction_files:
            # Skip the 'latest' and 'alerts' files as they were handled above
            if 'latest_prediction' not in pred_file and 'aqi_alerts' not in pred_file:
                filename = os.path.basename(pred_file)
                hf_path = f'predictions/{timestamp}_{filename}'
                
                api.upload_file(
                    path_or_fileobj=pred_file,
                    path_in_repo=hf_path,
                    repo_id=repo_id,
                    commit_message=f'🔮 Upload predictions: {timestamp}'
                )
                print(f'✅ Uploaded {pred_file} to Hugging Face Hub as {hf_path}')
        
        print("✅ Prediction upload process completed.")

    except Exception as e:
        print(f'❌ Prediction upload failed: {str(e)}')
        # You can choose to exit with an error code if this is a critical failure
        exit(1)

if __name__ == "__main__":
    main()
