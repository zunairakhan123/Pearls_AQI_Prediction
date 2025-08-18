import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import shap
import matplotlib.pyplot as plt
import re
import joblib
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

# This section dynamically gets the repo name and token from environment variables
HF_REPO = os.environ.get('HF_REPO')
HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_REPO:
    st.error("Hugging Face Repository not found. Please set the 'HF_REPO' environment variable in your Hugging Face Space settings.")
    st.stop()

# Set the HF token for authentication
os.environ["HF_TOKEN"] = HF_TOKEN

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP not available. Install with: pip install shap")

# Page configuration
st.set_page_config(
    page_title="Lahore AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
    }
    .alert-moderate {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ef6c00;
    }
    .daily-forecast {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class AQIDashboard:
    def __init__(self, hf_repo, hf_token):
        self.hf_repo = hf_repo
        self.hf_token = hf_token
        self.api = HfApi()
        
    @st.cache_resource
    def load_best_model_from_hub(_self):
        """Loads the best model from Hugging Face Hub based on lowest RMSE."""
        try:
            files = _self.api.list_repo_files(repo_id=_self.hf_repo, token=_self.hf_token, repo_type='model')
            model_files = [f for f in files if f.startswith('models/') and ('.joblib' in f or '.pkl' in f)]
            
            if not model_files:
                st.warning("No model files found in Hugging Face Hub.")
                return None, None, None

            best_model_file = None
            best_rmse = float('inf')
            
            for file_path in model_files:
                filename = os.path.basename(file_path)
                rmse_match = re.search(r'RMSE(\d+\.?\d*)', filename)
                if rmse_match:
                    rmse = float(rmse_match.group(1))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model_file = file_path
            
            if best_model_file:
                local_path = hf_hub_download(repo_id=_self.hf_repo, filename=best_model_file, token=_self.hf_token)
                model = joblib.load(local_path)
                model_info = {'model_name': os.path.basename(best_model_file), 'feature_columns': 'your_feature_columns_here'}
                return model, None, model_info 
            else:
                st.warning("No models with RMSE in filename found.")
                return None, None, None

        except Exception as e:
            st.error(f"Failed to load model from Hugging Face Hub: {e}")
            return None, None, None
    
    @st.cache_resource
    def load_latest_features_from_hub(_self):
        """Loads the latest feature file from Hugging Face Hub."""
        try:
            files = _self.api.list_repo_files(repo_id=_self.hf_repo, token=_self.hf_token, repo_type='model')
            feature_files = [f for f in files if f.startswith('features/') and 'features_' in f]
            
            if not feature_files:
                st.warning("No feature files found in Hugging Face Hub.")
                return pd.DataFrame()

            latest_feature_file = sorted(feature_files, reverse=True)[0]
            local_path = hf_hub_download(repo_id=_self.hf_repo, filename=latest_feature_file, token=_self.hf_token)
            
            features_df = pd.read_csv(local_path)
            return features_df

        except Exception as e:
            st.error(f"Failed to load features from Hugging Face Hub: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)  # Cache for 1 hour to prevent re-downloads
    def load_latest_predictions(_self):
        """Load latest predictions from Hugging Face Hub."""
        try:
            local_path = hf_hub_download(repo_id=_self.hf_repo, filename="predictions/latest_predictions.csv", token=_self.hf_token)
            return pd.read_csv(local_path)
        except Exception as e:
            st.warning(f"Failed to load latest predictions from Hugging Face Hub: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_alerts(_self):
        """Load AQI alerts from Hugging Face Hub."""
        try:
            local_path = hf_hub_download(repo_id=_self.hf_repo, filename="predictions/aqi_alerts.csv", token=_self.hf_token)
            return pd.read_csv(local_path)
        except Exception as e:
            st.warning(f"Failed to load alerts from Hugging Face Hub: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_historical_data(_self):
        """Load latest historical AQI data for visualization from Hugging Face Hub."""
        try:
            # Assumes the latest historical data is also pushed to the Hub
            local_path = hf_hub_download(repo_id=_self.hf_repo, filename="data/raw/aqi_data.csv", token=_self.hf_token)
            return pd.read_csv(local_path)
        except Exception as e:
            st.warning(f"Failed to load historical data from Hugging Face Hub: {e}")
            return pd.DataFrame()

    def get_aqi_color(self, aqi_value):
        """Get color based on AQI value"""
        if aqi_value <= 50:
            return "#00e400"
        elif aqi_value <= 100:
            return "#ffff00"
        elif aqi_value <= 150:
            return "#ff7e00"
        elif aqi_value <= 200:
            return "#ff0000"
        elif aqi_value <= 300:
            return "#8f3f97"
        else:
            return "#7e0023"
    
    def get_aqi_category(self, aqi_value):
        """Get AQI category name"""
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
    
    def create_daily_forecast_chart(self, prediction_data):
        if not all(key in prediction_data for key in ['predicted_aqi_24h', 'predicted_aqi_48h', 'predicted_aqi_72h']):
            return None
        days = ['Day 1 (24h)', 'Day 2 (48h)', 'Day 3 (72h)']
        values = [prediction_data['predicted_aqi_24h'], prediction_data['predicted_aqi_48h'], prediction_data['predicted_aqi_72h']]
        colors = [self.get_aqi_color(val) for val in values]
        fig = go.Figure(data=[go.Bar(x=days, y=values, marker_color=colors, text=[f'{val:.0f}' for val in values], textposition='auto', name='AQI Forecast')])
        fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
        fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate")
        fig.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy for Sensitive")
        fig.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Unhealthy")
        fig.update_layout(title="3-Day AQI Forecast", xaxis_title="Day", yaxis_title="AQI", height=400, showlegend=False)
        return fig
    
    def create_aqi_gauge(self, current_aqi, predicted_aqi_avg):
        fig = go.Figure()
        fig.add_trace(go.Indicator(mode="gauge+number+delta", value=current_aqi, domain={'x': [0, 0.5], 'y': [0, 1]}, title={'text': "Current AQI"}, delta={'reference': predicted_aqi_avg}, gauge={'axis': {'range': [None, 500]}, 'bar': {'color': self.get_aqi_color(current_aqi)}, 'steps': [{'range': [0, 50], 'color': "lightgray"}, {'range': [50, 100], 'color': "gray"}, {'range': [100, 150], 'color': "lightgray"}, {'range': [150, 200], 'color': "gray"}, {'range': [200, 300], 'color': "lightgray"}, {'range': [300, 500], 'color': "gray"}], 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 200}}))
        fig.add_trace(go.Indicator(mode="gauge+number", value=predicted_aqi_avg, domain={'x': [0.5, 1], 'y': [0, 1]}, title={'text': "Predicted AQI (3-day avg)"}, gauge={'axis': {'range': [None, 500]}, 'bar': {'color': self.get_aqi_color(predicted_aqi_avg)}, 'steps': [{'range': [0, 50], 'color': "lightgray"}, {'range': [50, 100], 'color': "gray"}, {'range': [100, 150], 'color': "lightgray"}, {'range': [150, 200], 'color': "gray"}, {'range': [200, 300], 'color': "lightgray"}, {'range': [300, 500], 'color': "gray"}], 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 200}}))
        fig.update_layout(height=400)
        return fig
    
    def create_trend_chart(self, historical_data):
        if historical_data.empty: return None
        try: historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], format='ISO8601')
        except (ValueError, TypeError):
            try: historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], format='mixed')
            except (ValueError, TypeError):
                try: historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], infer_datetime_format=True)
                except (ValueError, TypeError) as e:
                    st.error(f"Unable to parse timestamp data: {e}"); return None
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical_data['timestamp'], y=historical_data['aqi'], mode='lines+markers', name='AQI', line=dict(color='blue', width=2), marker=dict(size=4)))
        fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, annotation_text="Good", annotation_position="left")
        fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, annotation_text="Moderate", annotation_position="left")
        fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, annotation_text="Unhealthy for Sensitive", annotation_position="left")
        fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, annotation_text="Unhealthy", annotation_position="left")
        fig.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, annotation_text="Very Unhealthy", annotation_position="left")
        fig.add_hrect(y0=300, y1=500, fillcolor="maroon", opacity=0.1, annotation_text="Hazardous", annotation_position="left")
        fig.update_layout(title="AQI Trend (Last 7 Days)", xaxis_title="Time", yaxis_title="AQI", height=400, showlegend=True)
        return fig
    
    def show_daily_forecast_cards(self, latest_prediction):
        if not all(key in latest_prediction for key in ['predicted_aqi_24h', 'predicted_aqi_48h', 'predicted_aqi_72h']): return
        st.subheader("📅 3-Day Detailed Forecast")
        forecasts = [("Tomorrow (24h)", latest_prediction['predicted_aqi_24h']), ("Day 2 (48h)", latest_prediction['predicted_aqi_48h']), ("Day 3 (72h)", latest_prediction['predicted_aqi_72h'])]
        cols = st.columns(3)
        for i, (day_name, aqi_value) in enumerate(forecasts):
            with cols[i]:
                category = self.get_aqi_category(aqi_value)
                color = self.get_aqi_color(aqi_value)
                st.markdown(f"""
                <div class="daily-forecast">
                    <h4 style="margin: 0; color: {color};">{day_name}</h4>
                    <h2 style="margin: 0.5rem 0; color: {color};">{aqi_value:.0f}</h2>
                    <p style="margin: 0; font-weight: bold;">{category}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def run_dashboard(self):
        st.markdown('<div class="main-header">🌫️ Lahore AQI Predictor Dashboard</div>', unsafe_allow_html=True)
        st.sidebar.header("🔧 Controls")
        if st.sidebar.button("🔄 Update Predictions", type="primary"):
            st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
        
        predictions_df = self.load_latest_predictions()
        alerts_df = self.load_alerts()
        historical_data = self.load_historical_data()
        
        if predictions_df.empty:
            st.warning("No predictions available. The scheduled GitHub Action may not have run yet. Please check back in a few minutes.")
            return
        
        latest_prediction = predictions_df.iloc[-1]
        current_aqi = latest_prediction.get('current_aqi', 150)
        predicted_aqi_avg = latest_prediction['predicted_aqi_3day_avg']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric(label="Current AQI", value=f"{current_aqi:.0f}", delta=f"{self.get_aqi_category(current_aqi)}")
        with col2: st.metric(label="3-Day Average", value=f"{predicted_aqi_avg:.0f}", delta=f"{predicted_aqi_avg - current_aqi:+.0f}")
        with col3:
            if 'predicted_aqi_24h' in latest_prediction:
                st.metric(label="Tomorrow (24h)", value=f"{latest_prediction['predicted_aqi_24h']:.0f}", delta=f"{latest_prediction['predicted_aqi_24h'] - current_aqi:+.0f}")
        with col4:
            model_used = latest_prediction.get('model_used', 'Unknown')
            st.metric(label="Model Used", value=model_used.replace('_MultiOutput', ''))
        
        self.show_daily_forecast_cards(latest_prediction)
        st.subheader("📊 3-Day Forecast Chart")
        forecast_chart = self.create_daily_forecast_chart(latest_prediction)
        if forecast_chart: st.plotly_chart(forecast_chart, use_container_width=True)
        
        if not alerts_df.empty:
            st.subheader("🚨 Current Alerts")
            latest_alerts = alerts_df.tail(5)
            for _, alert in latest_alerts.iterrows():
                day_info = alert.get('day', 'All Days')
                if alert['level'] == 'HAZARDOUS': st.error(f"🚨 **{day_info}**: {alert['message']} - {alert['recommendation']}")
                elif alert['level'] in ['UNHEALTHY', 'CHANGE_ALERT', 'TREND_ALERT']: st.warning(f"⚠️ **{day_info}**: {alert['message']} - {alert['recommendation']}")
                else: st.info(f"ℹ️ **{day_info}**: {alert['message']} - {alert['recommendation']}")
        
        st.subheader("📊 AQI Overview")
        gauge_fig = self.create_aqi_gauge(current_aqi, predicted_aqi_avg)
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        if not historical_data.empty:
            st.subheader("📈 AQI Trend")
            trend_fig = self.create_trend_chart(historical_data.tail(168))
            if trend_fig: st.plotly_chart(trend_fig, use_container_width=True)
        
        st.subheader("🔮 Recent Predictions Comparison")
        if len(predictions_df) > 1:
            recent_predictions = predictions_df.tail(10)
            try: recent_predictions['timestamp'] = pd.to_datetime(recent_predictions['timestamp'], format='ISO8601')
            except (ValueError, TypeError):
                try: recent_predictions['timestamp'] = pd.to_datetime(recent_predictions['timestamp'], format='mixed')
                except (ValueError, TypeError):
                    try: recent_predictions['timestamp'] = pd.to_datetime(recent_predictions['timestamp'], infer_datetime_format=True)
                    except (ValueError, TypeError) as e:
                        st.warning(f"Unable to parse prediction timestamps: {e}")
                        recent_predictions = recent_predictions.reset_index(); x_axis = recent_predictions.index; x_title = "Prediction Index"
                    else: x_axis = recent_predictions['timestamp']; x_title = "Time"
                else: x_axis = recent_predictions['timestamp']; x_title = "Time"
            else: x_axis = recent_predictions['timestamp']; x_title = "Time"
            fig = go.Figure()
            if 'predicted_aqi_24h' in recent_predictions.columns: fig.add_trace(go.Scatter(x=x_axis, y=recent_predictions['predicted_aqi_24h'], mode='lines+markers', name='24h Predictions', line=dict(color='blue')))
            if 'predicted_aqi_48h' in recent_predictions.columns: fig.add_trace(go.Scatter(x=x_axis, y=recent_predictions['predicted_aqi_48h'], mode='lines+markers', name='48h Predictions', line=dict(color='orange')))
            if 'predicted_aqi_72h' in recent_predictions.columns: fig.add_trace(go.Scatter(x=x_axis, y=recent_predictions['predicted_aqi_72h'], mode='lines+markers', name='72h Predictions', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=x_axis, y=recent_predictions['predicted_aqi_3day_avg'], mode='lines+markers', name='3-Day Average', line=dict(color='red', dash='dash')))
            fig.update_layout(title="Recent Predictions History", xaxis_title=x_title, yaxis_title="AQI", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🤖 Model Information")
        st.write("Model performance data is dynamically loaded from Hugging Face Hub.")
        
        if SHAP_AVAILABLE: self.show_feature_importance()
        
        st.subheader("📋 Data Quality")
        col1, col2, col3 = st.columns(3)
        with col1:
            if not historical_data.empty:
                data_completeness = (1 - historical_data.isnull().sum().sum() / historical_data.size) * 100
                st.metric("Data Completeness", f"{data_completeness:.1f}%")
        with col2:
            if not predictions_df.empty:
                last_update = pd.to_datetime(predictions_df['timestamp'].iloc[-1])
                hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
                st.metric("Hours Since Last Update", f"{hours_since_update:.1f}")
        with col3: st.metric("Total Predictions Made", len(predictions_df))
    
    def show_feature_importance(self):
        st.subheader("🎯 Feature Importance (SHAP)")
        try:
            model, scaler, model_info = self.load_best_model_from_hub()
            if model is None: st.write("No trained model available for SHAP analysis."); return
            features_df = self.load_latest_features_from_hub()
            if features_df.empty: st.write("No feature data available for SHAP analysis."); return
            
            if 'feature_columns' in model_info and model_info['feature_columns']:
                feature_columns = model_info['feature_columns'].split(',')
                X = features_df[feature_columns].select_dtypes(include=[np.number])
            else:
                X = features_df.select_dtypes(include=[np.number]).drop(columns=['aqi'], errors='ignore')

            X = X.fillna(X.mean())
            sample_size = min(50, len(X))
            X_sample = X.sample(sample_size, random_state=42)
            if scaler is not None: X_sample = pd.DataFrame(scaler.transform(X_sample), columns=X_sample.columns)
            st.write("⚠️ SHAP analysis for multi-output models is complex. Showing feature importance for first output (24h prediction).")
            if 'RandomForest' in str(type(model)):
                try:
                    first_estimator = model.estimators_[0]
                    explainer = shap.TreeExplainer(first_estimator)
                    shap_values = explainer.shap_values(X_sample)
                    fig_shap = plt.figure(figsize=(10, 6)); shap.summary_plot(shap_values, X_sample, show=False); st.pyplot(fig_shap)
                except Exception as e:
                    st.write(f"TreeExplainer failed: {e}. Using KernelExplainer instead.")
                    def predict_first_output(X): return model.predict(X)[:, 0]
                    explainer = shap.KernelExplainer(predict_first_output, X_sample[:5])
                    shap_values = explainer.shap_values(X_sample[:10])
                    fig_shap = plt.figure(figsize=(10, 6)); shap.summary_plot(shap_values, X_sample[:10], show=False); st.pyplot(fig_shap)
            else:
                def predict_first_output(X): return model.predict(X)[:, 0]
                explainer = shap.KernelExplainer(predict_first_output, X_sample[:5])
                shap_values = explainer.shap_values(X_sample[:10])
                fig_shap = plt.figure(figsize=(10, 6)); shap.summary_plot(shap_values, X_sample[:10], show=False); st.pyplot(fig_shap)
        except Exception as e: st.write(f"SHAP analysis unavailable: {e}")

if __name__ == "__main__":
    dashboard = AQIDashboard(hf_repo=HF_REPO, hf_token=HF_TOKEN)
    dashboard.run_dashboard()
    