
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# from datetime import datetime, timedelta
# import os
# import shap
# import matplotlib.pyplot as plt
# import sys
# sys.path.append('..')

# # Try to import SHAP
# try:
#     import shap
#     SHAP_AVAILABLE = True
# except ImportError:
#     SHAP_AVAILABLE = False
#     st.warning("SHAP not available. Install with: pip install shap")

# from model_training.model_registry import ModelRegistry
# from pipelines.inference_pipeline import InferencePipeline

# # Page configuration
# st.set_page_config(
#     page_title="Lahore AQI Predictor",
#     page_icon="🌫️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #1f77b4;
#     }
#     .alert-high {
#         background-color: #ffebee;
#         color: #c62828;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #c62828;
#     }
#     .alert-moderate {
#         background-color: #fff3e0;
#         color: #ef6c00;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #ef6c00;
#     }
# </style>
# """, unsafe_allow_html=True)

# class AQIDashboard:
#     def __init__(self):
#         self.model_registry = ModelRegistry()
#         self.inference_pipeline = InferencePipeline()
        
#     def load_latest_predictions(self):
#         """Load latest predictions"""
#         predictions_file = "data/predictions/latest_predictions.csv"
#         if os.path.exists(predictions_file):
#             return pd.read_csv(predictions_file)
#         return pd.DataFrame()
    
#     def load_alerts(self):
#         """Load AQI alerts"""
#         alerts_file = "data/predictions/aqi_alerts.csv"
#         if os.path.exists(alerts_file):
#             return pd.read_csv(alerts_file)
#         return pd.DataFrame()
    
#     def load_historical_data(self):
#         """Load historical AQI data for visualization"""
#         import glob
        
#         # Try to load the most recent raw data
#         aqi_files = glob.glob("data/raw/aqi_data_*.csv")
#         if aqi_files:
#             latest_file = max(aqi_files, key=os.path.getmtime)
#             return pd.read_csv(latest_file)
#         return pd.DataFrame()
    
#     def get_aqi_color(self, aqi_value):
#         """Get color based on AQI value"""
#         if aqi_value <= 50:
#             return "#00e400"  # Good - Green
#         elif aqi_value <= 100:
#             return "#ffff00"  # Moderate - Yellow
#         elif aqi_value <= 150:
#             return "#ff7e00"  # Unhealthy for Sensitive - Orange
#         elif aqi_value <= 200:
#             return "#ff0000"  # Unhealthy - Red
#         elif aqi_value <= 300:
#             return "#8f3f97"  # Very Unhealthy - Purple
#         else:
#             return "#7e0023"  # Hazardous - Maroon
    
#     def get_aqi_category(self, aqi_value):
#         """Get AQI category name"""
#         if aqi_value <= 50:
#             return "Good"
#         elif aqi_value <= 100:
#             return "Moderate"
#         elif aqi_value <= 150:
#             return "Unhealthy for Sensitive Groups"
#         elif aqi_value <= 200:
#             return "Unhealthy"
#         elif aqi_value <= 300:
#             return "Very Unhealthy"
#         else:
#             return "Hazardous"
    
#     def create_aqi_gauge(self, current_aqi, predicted_aqi):
#         """Create AQI gauge chart"""
#         fig = go.Figure()
        
#         # Add current AQI gauge
#         fig.add_trace(go.Indicator(
#             mode = "gauge+number+delta",
#             value = current_aqi,
#             domain = {'x': [0, 0.5], 'y': [0, 1]},
#             title = {'text': "Current AQI"},
#             delta = {'reference': predicted_aqi},
#             gauge = {
#                 'axis': {'range': [None, 500]},
#                 'bar': {'color': self.get_aqi_color(current_aqi)},
#                 'steps': [
#                     {'range': [0, 50], 'color': "lightgray"},
#                     {'range': [50, 100], 'color': "gray"},
#                     {'range': [100, 150], 'color': "lightgray"},
#                     {'range': [150, 200], 'color': "gray"},
#                     {'range': [200, 300], 'color': "lightgray"},
#                     {'range': [300, 500], 'color': "gray"}
#                 ],
#                 'threshold': {
#                     'line': {'color': "red", 'width': 4},
#                     'thickness': 0.75,
#                     'value': 200
#                 }
#             }
#         ))
        
#         # Add predicted AQI gauge
#         fig.add_trace(go.Indicator(
#             mode = "gauge+number",
#             value = predicted_aqi,
#             domain = {'x': [0.5, 1], 'y': [0, 1]},
#             title = {'text': "Predicted AQI (3-day avg)"},
#             gauge = {
#                 'axis': {'range': [None, 500]},
#                 'bar': {'color': self.get_aqi_color(predicted_aqi)},
#                 'steps': [
#                     {'range': [0, 50], 'color': "lightgray"},
#                     {'range': [50, 100], 'color': "gray"},
#                     {'range': [100, 150], 'color': "lightgray"},
#                     {'range': [150, 200], 'color': "gray"},
#                     {'range': [200, 300], 'color': "lightgray"},
#                     {'range': [300, 500], 'color': "gray"}
#                 ],
#                 'threshold': {
#                     'line': {'color': "red", 'width': 4},
#                     'thickness': 0.75,
#                     'value': 200
#                 }
#             }
#         ))
        
#         fig.update_layout(height=400)
#         return fig
    
#     def create_trend_chart(self, historical_data):
#         """Create AQI trend chart"""
#         if historical_data.empty:
#             return None
            
#         # Convert timestamp to datetime
#         historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
        
#         # Create the trend chart
#         fig = go.Figure()
        
#         fig.add_trace(go.Scatter(
#             x=historical_data['timestamp'],
#             y=historical_data['aqi'],
#             mode='lines+markers',
#             name='AQI',
#             line=dict(color='blue', width=2),
#             marker=dict(size=4)
#         ))
        
#         # Add AQI category bands
#         fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, annotation_text="Good", annotation_position="left")
#         fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, annotation_text="Moderate", annotation_position="left")
#         fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, annotation_text="Unhealthy for Sensitive", annotation_position="left")
#         fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, annotation_text="Unhealthy", annotation_position="left")
#         fig.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, annotation_text="Very Unhealthy", annotation_position="left")
#         fig.add_hrect(y0=300, y1=500, fillcolor="maroon", opacity=0.1, annotation_text="Hazardous", annotation_position="left")
        
#         fig.update_layout(
#             title="AQI Trend (Last 7 Days)",
#             xaxis_title="Time",
#             yaxis_title="AQI",
#             height=400,
#             showlegend=True
#         )
        
#         return fig
    
#     def run_dashboard(self):
#         """Main dashboard function"""
#         # Header
#         st.markdown('<div class="main-header">🌫️ Lahore AQI Predictor Dashboard</div>', unsafe_allow_html=True)
        
#         # Sidebar
#         st.sidebar.header("🔧 Controls")
        
#         # Run inference button
#         if st.sidebar.button("🔄 Update Predictions", type="primary"):
#             with st.spinner("Running inference pipeline..."):
#                 result = self.inference_pipeline.run_inference_pipeline()
#                 if result:
#                     st.sidebar.success("Predictions updated!")
#                 else:
#                     st.sidebar.error("Failed to update predictions")
        
#         # Auto-refresh option
#         auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (every 5 minutes)")
#         if auto_refresh:
#             st.rerun()
        
#         # Load data
#         predictions_df = self.load_latest_predictions()
#         alerts_df = self.load_alerts()
#         historical_data = self.load_historical_data()
        
#         if predictions_df.empty:
#             st.warning("No predictions available. Please run the inference pipeline first.")
#             if st.button("Run Inference Pipeline"):
#                 with st.spinner("Running inference pipeline..."):
#                     result = self.inference_pipeline.run_inference_pipeline()
#                     if result:
#                         st.success("Predictions generated!")
#                         st.rerun()
#             return
        
#         # Get latest prediction
#         latest_prediction = predictions_df.iloc[-1]
#         current_aqi = latest_prediction.get('current_aqi', 150)  # Default if not available
#         predicted_aqi = latest_prediction['predicted_aqi_3day_avg']
        
#         # Main metrics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric(
#                 label="Current AQI",
#                 value=f"{current_aqi:.0f}",
#                 delta=f"{self.get_aqi_category(current_aqi)}"
#             )
        
#         with col2:
#             st.metric(
#                 label="Predicted AQI (3-day)",
#                 value=f"{predicted_aqi:.0f}",
#                 delta=f"{predicted_aqi - current_aqi:+.0f}"
#             )
        
#         with col3:
#             if 'predicted_aqi_24h' in latest_prediction:
#                 st.metric(
#                     label="24-hour Prediction",
#                     value=f"{latest_prediction['predicted_aqi_24h']:.0f}",
#                     delta=f"{self.get_aqi_category(latest_prediction['predicted_aqi_24h'])}"
#                 )
        
#         with col4:
#             model_used = latest_prediction.get('model_used', 'Unknown')
#             st.metric(
#                 label="Model Used",
#                 value=model_used
#             )
        
#         # Alerts section
#         if not alerts_df.empty:
#             st.subheader("🚨 Current Alerts")
#             latest_alerts = alerts_df.tail(3)
            
#             for _, alert in latest_alerts.iterrows():
#                 if alert['level'] == 'HAZARDOUS':
#                     st.error(f"🚨 {alert['message']} - {alert['recommendation']}")
#                 elif alert['level'] in ['UNHEALTHY', 'CHANGE_ALERT']:
#                     st.warning(f"⚠️ {alert['message']} - {alert['recommendation']}")
#                 else:
#                     st.info(f"ℹ️ {alert['message']} - {alert['recommendation']}")
        
#         # Gauge charts
#         st.subheader("📊 AQI Overview")
#         gauge_fig = self.create_aqi_gauge(current_aqi, predicted_aqi)
#         st.plotly_chart(gauge_fig, use_container_width=True)
        
#         # Trend chart
#         if not historical_data.empty:
#             st.subheader("📈 AQI Trend")
#             trend_fig = self.create_trend_chart(historical_data.tail(168))  # Last 7 days (hourly data)
#             if trend_fig:
#                 st.plotly_chart(trend_fig, use_container_width=True)
        
#         # Predictions history
#         st.subheader("🔮 Recent Predictions")
#         if len(predictions_df) > 1:
#             recent_predictions = predictions_df.tail(10)
#             recent_predictions['timestamp'] = pd.to_datetime(recent_predictions['timestamp'])
            
#             chart_data = recent_predictions[['timestamp', 'predicted_aqi_3day_avg']].copy()
#             chart_data = chart_data.set_index('timestamp')
            
#             st.line_chart(chart_data)
        
#         # Model performance
#         st.subheader("🤖 Model Information")
#         models_df = self.model_registry.list_all_models()
        
#         if not models_df.empty:
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.write("**Available Models:**")
#                 model_summary = models_df[['model_name', 'test_rmse', 'test_r2', 'created_at']].copy()
#                 model_summary = model_summary.sort_values('test_rmse')
#                 st.dataframe(model_summary)
            
#             with col2:
#                 st.write("**Model Performance Comparison:**")
#                 fig = px.bar(
#                     models_df, 
#                     x='model_name', 
#                     y='test_rmse',
#                     title="Model RMSE Comparison",
#                     color='test_rmse',
#                     color_continuous_scale='viridis_r'
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
        
#         # Feature importance (if SHAP is available)
#         if SHAP_AVAILABLE:
#             self.show_feature_importance()
        
#         # Data quality metrics
#         st.subheader("📋 Data Quality")
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             if not historical_data.empty:
#                 data_completeness = (1 - historical_data.isnull().sum().sum() / historical_data.size) * 100
#                 st.metric("Data Completeness", f"{data_completeness:.1f}%")
        
#         with col2:
#             if not predictions_df.empty:
#                 last_update = pd.to_datetime(predictions_df['timestamp'].iloc[-1])
#                 hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
#                 st.metric("Hours Since Last Update", f"{hours_since_update:.1f}")
        
#         with col3:
#             total_predictions = len(predictions_df)
#             st.metric("Total Predictions Made", total_predictions)
    
#     def show_feature_importance(self):
#         """Show SHAP feature importance if available"""
#         st.subheader("🎯 Feature Importance (SHAP)")
        
#         try:
#             # Load the best model
#             model, scaler, model_info = self.model_registry.get_best_model()
            
#             if model is None:
#                 st.write("No trained model available for SHAP analysis.")
#                 return
            
#             # Load latest features for SHAP analysis
#             from feature_store.feature_store_manager import FeatureStore
#             fs = FeatureStore()
#             features_df = fs.load_features_from_csv(latest=True)
            
#             if features_df is None or features_df.empty:
#                 st.write("No feature data available for SHAP analysis.")
#                 return
            
#             # Prepare features (same as training)
#             feature_columns = model_info['feature_columns'].split(',')
#             X = features_df[feature_columns].select_dtypes(include=[np.number])
#             X = X.fillna(X.mean())
            
#             # Take a sample for SHAP (to speed up computation)
#             sample_size = min(100, len(X))
#             X_sample = X.sample(sample_size, random_state=42)
            
#             # Scale if needed
#             if scaler is not None:
#                 X_sample = pd.DataFrame(scaler.transform(X_sample), columns=X_sample.columns)
            
#             # Create SHAP explainer
#             if model_info['model_name'] == 'RandomForest':
#                 explainer = shap.TreeExplainer(model)
#                 shap_values = explainer.shap_values(X_sample)
#             else:
#                 explainer = shap.KernelExplainer(model.predict, X_sample[:10])  # Use small background set
#                 shap_values = explainer.shap_values(X_sample[:20])  # Explain subset
            
#             # Create SHAP summary plot
#             fig_shap = plt.figure(figsize=(10, 6))
#             shap.summary_plot(shap_values, X_sample, show=False)
#             st.pyplot(fig_shap)
            
#         except Exception as e:
#             st.write(f"SHAP analysis unavailable: {e}")

# # Run the dashboard
# if __name__ == "__main__":
#     dashboard = AQIDashboard()
#     dashboard.run_dashboard()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import shap
import matplotlib.pyplot as plt
import sys
import glob

# Dynamically resolve project root and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add project root to Python path for imports
sys.path.append(project_root)

data_predictions_path = os.path.join(project_root, 'data', 'predictions')
data_raw_path = os.path.join(project_root, 'data', 'raw')

# Load latest predictions
predictions_file = os.path.join(data_predictions_path, 'latest_predictions.csv')
if os.path.exists(predictions_file):
    predictions_df = pd.read_csv(predictions_file)
    print(f"Loaded {len(predictions_df)} predictions")
else:
    print("No predictions file found.")

# Load AQI alerts
alerts_file = os.path.join(data_predictions_path, 'aqi_alerts.csv')
if os.path.exists(alerts_file):
    alerts_df = pd.read_csv(alerts_file)
    print(f"Loaded {len(alerts_df)} AQI alerts")
else:
    print("No AQI alerts file found.")

# Example: Load all raw AQI data files
aqis_files = glob.glob(os.path.join(data_raw_path, 'aqi_data_*.csv'))
if aqis_files:
    aqi_data_list = [pd.read_csv(file) for file in aqis_files]
    combined_aqi_data = pd.concat(aqi_data_list, ignore_index=True)
    print(f"Loaded {len(combined_aqi_data)} AQI data rows from {len(aqis_files)} files")
else:
    print("No AQI raw data files found.")

# Example: Load all raw weather data files
weather_files = glob.glob(os.path.join(data_raw_path, 'weather_data_*.csv'))
if weather_files:
    weather_data_list = [pd.read_csv(file) for file in weather_files]
    combined_weather_data = pd.concat(weather_data_list, ignore_index=True)
    print(f"Loaded {len(combined_weather_data)} weather data rows from {len(weather_files)} files")
else:
    print("No weather raw data files found.")

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP not available. Install with: pip install shap")

# Import modules with dynamic paths
try:
    from model_training.model_registry import ModelRegistry
    from pipelines.inference_pipeline import InferencePipeline
except ImportError as e:
    st.error(f"Could not import required modules: {e}")
    st.stop()

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
    def __init__(self):
        self.project_root = project_root
        self.data_predictions_path = data_predictions_path
        self.data_raw_path = data_raw_path
        
        self.model_registry = ModelRegistry()
        self.inference_pipeline = InferencePipeline()
        
    def load_latest_predictions(self):
        """Load latest predictions"""
        predictions_file = os.path.join(self.data_predictions_path, "latest_predictions.csv")
        if os.path.exists(predictions_file):
            return pd.read_csv(predictions_file)
        return pd.DataFrame()
    
    def load_alerts(self):
        """Load AQI alerts"""
        alerts_file = os.path.join(self.data_predictions_path, "aqi_alerts.csv")
        if os.path.exists(alerts_file):
            return pd.read_csv(alerts_file)
        return pd.DataFrame()
    
    def load_historical_data(self):
        """Load historical AQI data for visualization"""
        # Try to load the most recent raw data
        aqi_files = glob.glob(os.path.join(self.data_raw_path, "aqi_data_*.csv"))
        if aqi_files:
            latest_file = max(aqi_files, key=os.path.getmtime)
            return pd.read_csv(latest_file)
        return pd.DataFrame()
    
    def get_aqi_color(self, aqi_value):
        """Get color based on AQI value"""
        if aqi_value <= 50:
            return "#00e400"  # Good - Green
        elif aqi_value <= 100:
            return "#ffff00"  # Moderate - Yellow
        elif aqi_value <= 150:
            return "#ff7e00"  # Unhealthy for Sensitive - Orange
        elif aqi_value <= 200:
            return "#ff0000"  # Unhealthy - Red
        elif aqi_value <= 300:
            return "#8f3f97"  # Very Unhealthy - Purple
        else:
            return "#7e0023"  # Hazardous - Maroon
    
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
        """Create daily forecast chart for 3 individual days"""
        if not all(key in prediction_data for key in ['predicted_aqi_24h', 'predicted_aqi_48h', 'predicted_aqi_72h']):
            return None
            
        days = ['Day 1 (24h)', 'Day 2 (48h)', 'Day 3 (72h)']
        values = [
            prediction_data['predicted_aqi_24h'],
            prediction_data['predicted_aqi_48h'], 
            prediction_data['predicted_aqi_72h']
        ]
        colors = [self.get_aqi_color(val) for val in values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=days,
                y=values,
                marker_color=colors,
                text=[f'{val:.0f}' for val in values],
                textposition='auto',
                name='AQI Forecast'
            )
        ])
        
        # Add AQI category lines
        fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
        fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate")
        fig.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy for Sensitive")
        fig.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Unhealthy")
        
        fig.update_layout(
            title="3-Day AQI Forecast",
            xaxis_title="Day",
            yaxis_title="AQI",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_aqi_gauge(self, current_aqi, predicted_aqi_avg):
        """Create AQI gauge chart"""
        fig = go.Figure()
        
        # Add current AQI gauge
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = current_aqi,
            domain = {'x': [0, 0.5], 'y': [0, 1]},
            title = {'text': "Current AQI"},
            delta = {'reference': predicted_aqi_avg},
            gauge = {
                'axis': {'range': [None, 500]},
                'bar': {'color': self.get_aqi_color(current_aqi)},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"},
                    {'range': [100, 150], 'color': "lightgray"},
                    {'range': [150, 200], 'color': "gray"},
                    {'range': [200, 300], 'color': "lightgray"},
                    {'range': [300, 500], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 200
                }
            }
        ))
        
        # Add predicted AQI gauge
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = predicted_aqi_avg,
            domain = {'x': [0.5, 1], 'y': [0, 1]},
            title = {'text': "Predicted AQI (3-day avg)"},
            gauge = {
                'axis': {'range': [None, 500]},
                'bar': {'color': self.get_aqi_color(predicted_aqi_avg)},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"},
                    {'range': [100, 150], 'color': "lightgray"},
                    {'range': [150, 200], 'color': "gray"},
                    {'range': [200, 300], 'color': "lightgray"},
                    {'range': [300, 500], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 200
                }
            }
        ))
        
        fig.update_layout(height=400)
        return fig
    
    def create_trend_chart(self, historical_data):
        """Create AQI trend chart"""
        if historical_data.empty:
            return None
            
        # Convert timestamp to datetime
        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
        
        # Create the trend chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical_data['timestamp'],
            y=historical_data['aqi'],
            mode='lines+markers',
            name='AQI',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add AQI category bands
        fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, annotation_text="Good", annotation_position="left")
        fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, annotation_text="Moderate", annotation_position="left")
        fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, annotation_text="Unhealthy for Sensitive", annotation_position="left")
        fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, annotation_text="Unhealthy", annotation_position="left")
        fig.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, annotation_text="Very Unhealthy", annotation_position="left")
        fig.add_hrect(y0=300, y1=500, fillcolor="maroon", opacity=0.1, annotation_text="Hazardous", annotation_position="left")
        
        fig.update_layout(
            title="AQI Trend (Last 7 Days)",
            xaxis_title="Time",
            yaxis_title="AQI",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def show_daily_forecast_cards(self, latest_prediction):
        """Display daily forecast as cards"""
        if not all(key in latest_prediction for key in ['predicted_aqi_24h', 'predicted_aqi_48h', 'predicted_aqi_72h']):
            return
            
        st.subheader("📅 3-Day Detailed Forecast")
        
        forecasts = [
            ("Tomorrow (24h)", latest_prediction['predicted_aqi_24h']),
            ("Day 2 (48h)", latest_prediction['predicted_aqi_48h']),
            ("Day 3 (72h)", latest_prediction['predicted_aqi_72h'])
        ]
        
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
        """Main dashboard function"""
        # Header
        st.markdown('<div class="main-header">🌫️ Lahore AQI Predictor Dashboard</div>', unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.header("🔧 Controls")
        
        # Run inference button
        if st.sidebar.button("🔄 Update Predictions", type="primary"):
            with st.spinner("Running inference pipeline..."):
                result = self.inference_pipeline.run_inference_pipeline()
                if result:
                    st.sidebar.success("Predictions updated!")
                else:
                    st.sidebar.error("Failed to update predictions")
        
        # Auto-refresh option
        auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (every 5 minutes)")
        if auto_refresh:
            st.rerun()
        
        # Load data
        predictions_df = self.load_latest_predictions()
        alerts_df = self.load_alerts()
        historical_data = self.load_historical_data()
        
        if predictions_df.empty:
            st.warning("No predictions available. Please run the inference pipeline first.")
            if st.button("Run Inference Pipeline"):
                with st.spinner("Running inference pipeline..."):
                    result = self.inference_pipeline.run_inference_pipeline()
                    if result:
                        st.success("Predictions generated!")
                        st.rerun()
            return
        
        # Get latest prediction
        latest_prediction = predictions_df.iloc[-1]
        current_aqi = latest_prediction.get('current_aqi', 150)  # Default if not available
        predicted_aqi_avg = latest_prediction['predicted_aqi_3day_avg']
        
        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Current AQI",
                value=f"{current_aqi:.0f}",
                delta=f"{self.get_aqi_category(current_aqi)}"
            )
        
        with col2:
            st.metric(
                label="3-Day Average",
                value=f"{predicted_aqi_avg:.0f}",
                delta=f"{predicted_aqi_avg - current_aqi:+.0f}"
            )
        
        with col3:
            if 'predicted_aqi_24h' in latest_prediction:
                st.metric(
                    label="Tomorrow (24h)",
                    value=f"{latest_prediction['predicted_aqi_24h']:.0f}",
                    delta=f"{latest_prediction['predicted_aqi_24h'] - current_aqi:+.0f}"
                )
        
        with col4:
            model_used = latest_prediction.get('model_used', 'Unknown')
            st.metric(
                label="Model Used",
                value=model_used.replace('_MultiOutput', '')
            )
        
        # Daily forecast cards
        self.show_daily_forecast_cards(latest_prediction)
        
        # Daily forecast chart
        st.subheader("📊 3-Day Forecast Chart")
        forecast_chart = self.create_daily_forecast_chart(latest_prediction)
        if forecast_chart:
            st.plotly_chart(forecast_chart, use_container_width=True)
        
        # Alerts section
        if not alerts_df.empty:
            st.subheader("🚨 Current Alerts")
            latest_alerts = alerts_df.tail(5)
            
            for _, alert in latest_alerts.iterrows():
                day_info = alert.get('day', 'All Days')
                if alert['level'] == 'HAZARDOUS':
                    st.error(f"🚨 **{day_info}**: {alert['message']} - {alert['recommendation']}")
                elif alert['level'] in ['UNHEALTHY', 'CHANGE_ALERT', 'TREND_ALERT']:
                    st.warning(f"⚠️ **{day_info}**: {alert['message']} - {alert['recommendation']}")
                else:
                    st.info(f"ℹ️ **{day_info}**: {alert['message']} - {alert['recommendation']}")
        
        # Gauge charts
        st.subheader("📊 AQI Overview")
        gauge_fig = self.create_aqi_gauge(current_aqi, predicted_aqi_avg)
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Trend chart
        if not historical_data.empty:
            st.subheader("📈 AQI Trend")
            trend_fig = self.create_trend_chart(historical_data.tail(168))  # Last 7 days (hourly data)
            if trend_fig:
                st.plotly_chart(trend_fig, use_container_width=True)
        
        # Predictions history
        st.subheader("🔮 Recent Predictions Comparison")
        if len(predictions_df) > 1:
            recent_predictions = predictions_df.tail(10)
            recent_predictions['timestamp'] = pd.to_datetime(recent_predictions['timestamp'])
            
            # Multi-line chart for different predictions
            fig = go.Figure()
            
            if 'predicted_aqi_24h' in recent_predictions.columns:
                fig.add_trace(go.Scatter(
                    x=recent_predictions['timestamp'],
                    y=recent_predictions['predicted_aqi_24h'],
                    mode='lines+markers',
                    name='24h Predictions',
                    line=dict(color='blue')
                ))
            
            if 'predicted_aqi_48h' in recent_predictions.columns:
                fig.add_trace(go.Scatter(
                    x=recent_predictions['timestamp'],
                    y=recent_predictions['predicted_aqi_48h'],
                    mode='lines+markers',
                    name='48h Predictions',
                    line=dict(color='orange')
                ))
            
            if 'predicted_aqi_72h' in recent_predictions.columns:
                fig.add_trace(go.Scatter(
                    x=recent_predictions['timestamp'],
                    y=recent_predictions['predicted_aqi_72h'],
                    mode='lines+markers',
                    name='72h Predictions',
                    line=dict(color='green')
                ))
            
            fig.add_trace(go.Scatter(
                x=recent_predictions['timestamp'],
                y=recent_predictions['predicted_aqi_3day_avg'],
                mode='lines+markers',
                name='3-Day Average',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="Recent Predictions History",
                xaxis_title="Time",
                yaxis_title="AQI",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model performance
        st.subheader("🤖 Model Information")
        models_df = self.model_registry.list_all_models()
        
        if not models_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Available Models:**")
                # Updated to handle multi-output model metrics
                display_columns = ['model_name', 'created_at']
                
                # Add RMSE column (prefer overall, fallback to test_rmse)
                if 'test_overall_rmse' in models_df.columns:
                    display_columns.append('test_overall_rmse')
                    models_df = models_df.rename(columns={'test_overall_rmse': 'RMSE'})
                elif 'test_rmse' in models_df.columns:
                    display_columns.append('test_rmse')
                    models_df = models_df.rename(columns={'test_rmse': 'RMSE'})
                
                # Add R2 column
                if 'test_overall_r2' in models_df.columns:
                    display_columns.append('test_overall_r2')
                    models_df = models_df.rename(columns={'test_overall_r2': 'R²'})
                elif 'test_r2' in models_df.columns:
                    display_columns.append('test_r2')
                    models_df = models_df.rename(columns={'test_r2': 'R²'})
                
                # Update display_columns to use renamed columns
                display_columns = [col if col in ['model_name', 'created_at'] else 
                                 ('RMSE' if 'rmse' in col.lower() else 'R²' if 'r2' in col.lower() else col) 
                                 for col in display_columns]
                
                available_display_cols = [col for col in display_columns if col in models_df.columns]
                model_summary = models_df[available_display_cols].copy()
                
                if 'RMSE' in model_summary.columns:
                    model_summary = model_summary.sort_values('RMSE')
                    
                st.dataframe(model_summary)
            
            with col2:
                st.write("**Model Performance Comparison:**")
                # Use appropriate RMSE column for comparison
                rmse_col = 'test_overall_rmse' if 'test_overall_rmse' in models_df.columns else 'test_rmse'
                
                if rmse_col in models_df.columns:
                    fig = px.bar(
                        models_df, 
                        x='model_name', 
                        y=rmse_col,
                        title="Model RMSE Comparison",
                        color=rmse_col,
                        color_continuous_scale='viridis_r'
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if SHAP is available)
        if SHAP_AVAILABLE:
            self.show_feature_importance()
        
        # Data quality metrics
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
        
        with col3:
            total_predictions = len(predictions_df)
            st.metric("Total Predictions Made", total_predictions)
    
    def show_feature_importance(self):
        """Show SHAP feature importance if available"""
        st.subheader("🎯 Feature Importance (SHAP)")
        
        try:
            # Load the best model (updated for multi-output)
            model, scaler, model_info = self.model_registry.get_best_model(metric='test_overall_rmse')
            
            if model is None:
                st.write("No trained model available for SHAP analysis.")
                return
            
            # Load latest features for SHAP analysis - use dynamic path
            try:
                from feature_store.feature_store_manager import FeatureStore
                fs = FeatureStore()
                features_df = fs.load_features_from_csv(latest=True)
            except ImportError:
                st.write("Feature store not available for SHAP analysis.")
                return
            
            if features_df is None or features_df.empty:
                st.write("No feature data available for SHAP analysis.")
                return
            
            # Prepare features (same as training)
            feature_columns = model_info['feature_columns'].split(',')
            X = features_df[feature_columns].select_dtypes(include=[np.number])
            X = X.fillna(X.mean())
            
            # Take a sample for SHAP (to speed up computation)
            sample_size = min(50, len(X))
            X_sample = X.sample(sample_size, random_state=42)
            
            # Scale if needed
            if scaler is not None:
                X_sample = pd.DataFrame(scaler.transform(X_sample), columns=X_sample.columns)
            
            # For multi-output models, we need to handle SHAP differently
            st.write("⚠️ SHAP analysis for multi-output models is complex. Showing feature importance for first output (24h prediction).")
            
            # Create SHAP explainer
            if 'RandomForest' in model_info['model_name']:
                # For multi-output RF, we need to access individual estimators
                try:
                    # Get the first estimator (for 24h prediction)
                    first_estimator = model.estimators_[0]
                    explainer = shap.TreeExplainer(first_estimator)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Create SHAP summary plot
                    fig_shap = plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_sample, show=False)
                    st.pyplot(fig_shap)
                    
                except Exception as e:
                    st.write(f"TreeExplainer failed: {e}. Using KernelExplainer instead.")
                    # Fallback to kernel explainer
                    def predict_first_output(X):
                        return model.predict(X)[:, 0]  # Only first output
                    
                    explainer = shap.KernelExplainer(predict_first_output, X_sample[:5])
                    shap_values = explainer.shap_values(X_sample[:10])
                    
                    fig_shap = plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_sample[:10], show=False)
                    st.pyplot(fig_shap)
            else:
                # For other multi-output models
                def predict_first_output(X):
                    return model.predict(X)[:, 0]  # Only first output
                
                explainer = shap.KernelExplainer(predict_first_output, X_sample[:5])
                shap_values = explainer.shap_values(X_sample[:10])
                
                fig_shap = plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample[:10], show=False)
                st.pyplot(fig_shap)
            
        except Exception as e:
            st.write(f"SHAP analysis unavailable: {e}")

# Run the dashboard
if __name__ == "__main__":
    dashboard = AQIDashboard()
    dashboard.run_dashboard()