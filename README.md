# 🌫️ Pearls AQI Predictor

A fully automated Air Quality Index (AQI) prediction system for Lahore, Pakistan, built using only open-source tools and technologies.

## 🎯 Overview

This system provides:
- **Real-time AQI monitoring** for Lahore city
- **3-day AQI predictions** using machine learning
- **Automated data pipeline** with hourly updates
- **Interactive dashboard** with SHAP explanations
- **Hazardous air quality alerts**
- **Model performance tracking**

## 📂 Project Structure

```
Pearls_AQIPredictor/
│
├── data/
│   ├── raw/              # Raw AQI and weather data
│   ├── features/         # Processed features
│   └── predictions/      # Model predictions
│
├── models/               # Trained models storage
│
├── data_fetching/
│   └── fetch_aqi_data.py           # Data fetching from APIs
│
├── feature_engineering/
│   └── compute_features.py         # Feature computation
│
├── feature_store/
│   └── feature_store_manager.py    # CSV-based feature store
│
├── model_training/
│   ├── train_model.py              # Model training pipeline
│   └── model_registry.py           # Model management
│
├── pipelines/
│   ├── backfill_pipeline.py        # Historical data processing
│   └── inference_pipeline.py       # Real-time predictions
│
├── dashboard/
│   └── app.py                      # Streamlit dashboard
│
├── cicd/
│   ├── github_actions.yml          # GitHub Actions CI/CD
│   └── airflow_dag.py              # Airflow DAG
│
├── explainability/
│   └── shap_explain.py             # Model explainability
│
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Pearls_AQIPredictor.git
cd Pearls_AQIPredictor

# Install dependencies
pip install pandas numpy scikit-learn joblib streamlit plotly shap requests
```

### 2. API Keys Setup

Create a `.env` file in the root directory:

```bash
# Free API keys (replace with actual keys)
AQI_API_KEY=your_aqicn_api_key_here
WEATHER_API_KEY=your_openweather_api_key_here
```

**Free API Sources:**
- [AQICN API](https://aqicn.org/api/) - Free tier: 1000 requests/day
- [OpenWeather API](https://openweathermap.org/api) - Free tier: 1000 requests/day

### 3. Initial Setup & Backfill

```bash
# Run the backfill pipeline (fetches data from July 1st)
python pipelines/backfill_pipeline.py
```

This will:
- Fetch historical AQI and weather data
- Compute features
- Train initial models
- Save everything to CSV files

### 4. Start the Dashboard

```bash
# Launch Streamlit dashboard
streamlit run dashboard/app.py
```

Visit `http://localhost:8501` to view the dashboard.

## 🔄 Automated Pipelines

### Hourly Data Processing
```bash
# Fetch latest data and make predictions
python pipelines/inference_pipeline.py
```

### Daily Model Training
```bash
# Retrain models with latest data
python model_training/train_model.py
```

### SHAP Explanations
```bash
# Generate model explanations
python explainability/shap_explain.py
```

## 🤖 CI/CD Automation

### GitHub Actions Setup

1. **Enable GitHub Actions** in your repository settings
2. **Add API keys** as repository secrets:
   - `AQI_API_KEY`
   - `WEATHER_API_KEY`
3. **Push to main branch** to trigger the workflow

The GitHub Actions workflow will:
- ✅ Run **every hour** for data fetching and predictions
- ✅ Run **daily at 6 AM** for model training
- ✅ Run **tests** on every push/PR
- ✅ **Auto-commit** updated data and models

### Airflow Setup (Optional)

If you prefer Apache Airflow:

```bash
# Install Airflow
pip install apache-airflow

# Initialize Airflow
airflow db init

# Copy DAG file
cp cicd/airflow_dag.py ~/airflow/dags/

# Start Airflow
airflow webserver --port 8080
airflow scheduler
```

## 📊 Dashboard Features

The Streamlit dashboard provides:

### 🎛️ Real-time Monitoring
- Current AQI gauge with color-coded categories
- 3-day prediction gauge
- AQI trend visualization
- Air quality category indicators

### 🚨 Smart Alerts
- **Hazardous** (AQI > 200): Red alerts with safety recommendations
- **Unhealthy** (AQI > 150): Orange warnings
- **Moderate** (AQI > 100): Yellow notifications
- **Rapid changes**: Alerts for sudden AQI shifts

### 🤖 Model Performance
- Model comparison metrics (RMSE, MAE, R²)
- Feature importance rankings
- SHAP value explanations
- Model training history

### 📈 Data Insights
- Historical AQI trends
- Prediction accuracy tracking
- Data quality metrics
- Feature correlation analysis

## 🧠 Machine Learning Models

The system trains and compares three models:

### 1. **Random Forest** (Primary)
- Handles non-linear relationships
- Feature importance built-in
- Robust to outliers
- Best for air quality patterns

### 2. **Ridge Regression** (Baseline)
- Simple linear model
- Fast training and inference
- Good interpretability
- Regularization prevents overfitting

### 3. **Neural Network** (Advanced)
- Captures complex patterns
- Non-linear transformations
- Handles feature interactions
- Can model temporal dependencies

**Model Selection:** The system automatically uses the model with the lowest test RMSE.

## 🎯 Features Engineering

The system creates 50+ features including:

### Time-based Features
- Hour of day, day of week, month
- Weekend/weekday indicators
- Seasonal patterns

### Lag Features
- AQI values from 1, 6, 12, 24 hours ago
- Weather parameters from previous hours
- Moving averages and trends

### Derived Features
- AQI change rate
- PM2.5/PM10 ratio
- Heat index (temperature + humidity)
- Air quality categories

### Rolling Statistics
- 6-hour, 12-hour, 24-hour rolling means
- Rolling standard deviation
- Rolling min/max values

## 🔍 Model Explainability

### SHAP (SHapley Additive exPlanations)
- **Feature importance**: Which factors matter most
- **Individual predictions**: Why a specific prediction was made
- **Model behavior**: How features interact
- **Trust and transparency**: Understand model decisions

### Visualizations
- Summary plots showing feature impact
- Waterfall plots for individual predictions
- Feature importance bar charts
- Partial dependence plots

## 📈 Performance Metrics

The system tracks:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of determination
- **Data completeness**: Percentage of non-missing data
- **Prediction latency**: Time to generate predictions
- **Model drift**: Performance degradation over time

## 🛠️ Troubleshooting

### Common Issues

**1. "No data found" error:**
```bash
# Run backfill first
python pipelines/backfill_pipeline.py
```

**2. "No trained model" error:**
```bash
# Train models manually
python model_training/train_model.py
```

**3. API rate limit exceeded:**
```bash
# Check your API keys and usage limits
# Reduce fetching frequency if needed
```

**4. SHAP installation issues:**
```bash
# Install SHAP separately
pip install shap
# Or use conda
conda install -c conda-forge shap
```

### Performance Optimization

**For large datasets:**
- Reduce `sample_size` in SHAP explanations
- Use feature selection to reduce dimensionality
- Implement data sampling for training

**For faster inference:**
- Use Ridge regression for speed
- Cache feature computations
- Implement feature importance filtering

## 🔒 Data Privacy & Security

- **No personal data**: Only environmental measurements
- **Local storage**: All data stored in CSV files locally
- **Open source**: Full transparency of data processing
- **API security**: Use environment variables for API keys

## 🌍 Environmental Impact

This system helps:
- **Public health**: Early warnings for harmful air quality
- **Environmental awareness**: Real-time air quality tracking  
- **Data-driven decisions**: Evidence-based air quality policies
- **Community engagement**: Accessible air quality information

## 🤝 Contributing

We welcome contributions! Please:

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** tests for new functionality
4. **Submit** a pull request

### Development Setup

```bash
# Install development dependencies
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Format code
black .

# Check code style
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AQICN.org** for providing free AQI data
- **OpenWeather** for weather data API
- **Streamlit** for the dashboard framework
- **SHAP** for model explainability
- **scikit-learn** for machine learning tools

## 📞 Support

For questions, issues, or suggestions:
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/Pearls_AQIPredictor/issues)
- **Email**: your.email@example.com
- **Documentation**: Check this README and code comments

---

**⚡ Built with ❤️ for cleaner air in Lahore, Pakistan**

*This system uses only open-source tools and free-tier APIs to ensure accessibility and sustainability.*
