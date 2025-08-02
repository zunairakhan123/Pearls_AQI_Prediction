import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Dynamic Project Root Resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# SHAP Import Check
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

from model_training.model_registry import ModelRegistry
from feature_store.feature_store_manager import FeatureStore

class SHAPExplainer:
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.feature_store = FeatureStore()
        self.explainability_path = os.path.join(project_root, "explainability")
        os.makedirs(self.explainability_path, exist_ok=True)
        
    def load_model_and_data(self):
        model, scaler, model_info = self.model_registry.get_best_model(metric='test_rmse')
        if model is None:
            print("No trained model found.")
            return None, None, None, None
        
        features_df = self.feature_store.load_features_from_csv(latest=True)
        if features_df is None:
            print("No feature data found.")
            return None, None, None, None
        
        feature_columns = model_info['feature_columns'].split(',')
        X = features_df[feature_columns].select_dtypes(include=[np.number]).fillna(0)
        
        if scaler is not None:
            X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        print(f"Loaded model: {model_info['model_name']} | Feature data shape: {X.shape}")
        return model, X_scaled, X, model_info
    
    def create_tree_explainer(self, model, X_sample):
        if not SHAP_AVAILABLE:
            return None, None
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            return explainer, shap_values
        except Exception as e:
            print(f"Error creating tree explainer: {e}")
            return None, None
    
    def create_kernel_explainer(self, model, X_background, X_sample):
        if not SHAP_AVAILABLE:
            return None, None
        
        try:
            background = X_background.sample(min(50, len(X_background)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, background)
            sample = X_sample.sample(min(20, len(X_sample)), random_state=42)
            shap_values = explainer.shap_values(sample)
            return explainer, shap_values
        except Exception as e:
            print(f"Error creating kernel explainer: {e}")
            return None, None
    
    def generate_summary_plot(self, shap_values, X_sample):
        if not SHAP_AVAILABLE:
            return None
        
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            summary_plot_path = os.path.join(self.explainability_path, "shap_summary.png")
            plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Summary plot saved to {summary_plot_path}")
            return summary_plot_path
        except Exception as e:
            print(f"Error generating summary plot: {e}")
            return None
    
    def generate_waterfall_plot(self, explainer, shap_values, X_sample, instance_idx=0):
        if not SHAP_AVAILABLE:
            return None
        
        try:
            plt.figure(figsize=(12, 8))
            if hasattr(shap, 'waterfall_plot'):
                shap.waterfall_plot(
                    explainer.expected_value, 
                    shap_values[instance_idx], 
                    X_sample.iloc[instance_idx], 
                    show=False
                )
            else:
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[instance_idx],
                    X_sample.iloc[instance_idx],
                    matplotlib=True,
                    show=False
                )
            waterfall_plot_path = os.path.join(self.explainability_path, "shap_waterfall.png")
            plt.savefig(waterfall_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Waterfall plot saved to {waterfall_plot_path}")
            return waterfall_plot_path
        except Exception as e:
            print(f"Error generating waterfall plot: {e}")
            return None
    
    def generate_feature_importance_bar(self, shap_values, X_sample):
        if not SHAP_AVAILABLE:
            return None
        
        try:
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            feature_importance = pd.DataFrame({
                'feature': X_sample.columns,
                'importance': mean_shap_values
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(10, 8))
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.xlabel('Mean |SHAP value|')
            plt.title('Feature Importance (SHAP)')
            plt.tight_layout()
            importance_plot_path = os.path.join(self.explainability_path, "feature_importance.png")
            plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            importance_csv_path = os.path.join(self.explainability_path, "feature_importance.csv")
            feature_importance.to_csv(importance_csv_path, index=False)
            print(f"Feature importance saved to {importance_plot_path}")
            return importance_plot_path
        except Exception as e:
            print(f"Error generating feature importance plot: {e}")
            return None
    
    def explain_model(self, sample_size=100, save_plots=True):
        print("Starting SHAP model explanation...")
        if not SHAP_AVAILABLE:
            print("SHAP library not available. Install with: pip install shap")
            return None
        
        model, X_scaled, X_original, model_info = self.load_model_and_data()
        if model is None:
            return None
        
        sample_size = min(sample_size, len(X_scaled))
        X_sample = X_scaled.sample(sample_size, random_state=42)
        X_sample_original = X_original.loc[X_sample.index]
        
        if model_info['model_name'] == 'RandomForest':
            explainer, shap_values = self.create_tree_explainer(model, X_sample)
        else:
            explainer, shap_values = self.create_kernel_explainer(model, X_scaled, X_sample)
        
        if explainer is None or shap_values is None:
            print("Failed to create SHAP explainer")
            return None
        
        results = {
            'model_name': model_info['model_name'],
            'sample_size': len(X_sample),
            'feature_count': len(X_sample.columns)
        }
        
        if save_plots:
            results['summary_plot'] = self.generate_summary_plot(shap_values, X_sample_original)
            results['waterfall_plot'] = self.generate_waterfall_plot(explainer, shap_values, X_sample_original)
            results['importance_plot'] = self.generate_feature_importance_bar(shap_values, X_sample_original)
        
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance_score': mean_shap_values
        }).sort_values('importance_score', ascending=False)
        
        results['feature_importance'] = feature_importance
        
        print("SHAP explanation completed successfully!")
        print("Top 5 Important Features:")
        print(feature_importance.head())
        
        return results

if __name__ == "__main__":
    explainer = SHAPExplainer()
    results = explainer.explain_model()
    
    if results:
        print(f"Explanation completed for {results['model_name']} model")
        print(f"Analyzed {results['sample_size']} samples with {results['feature_count']} features")
