#synthetic_aqi_dat.py
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --------------------------
# Paths
# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

weather_file = os.path.join(RAW_DIR, "weather_data_scraped.csv")
aqi_file = os.path.join(RAW_DIR, "aqi_data_synthetic_new.csv")

# --------------------------
# Load weather data
# --------------------------
df = pd.read_csv(weather_file)

required_columns = ["timestamp", "temperature", "humidity", "pressure", "wind_speed"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column in weather data: {col}")

# Ensure timestamp is datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --------------------------
# Enhanced PM generation function
# --------------------------
def generate_pm25_pm10(row):
    month = row["timestamp"].month
    hour = row["timestamp"].hour
    wind_speed = row["wind_speed"]
    humidity = row["humidity"]
    temperature = row["temperature"]
    
    # Seasonal baselines (Lahore patterns)
    if month in [12, 1, 2]:  # Winter - High pollution
        base_pm25 = np.random.normal(180, 25)
    elif month in [7, 8, 9]:  # Monsoon - Low pollution  
        base_pm25 = np.random.normal(65, 12)
    elif month in [3, 4, 5]:  # Pre-summer dust season
        base_pm25 = np.random.normal(165, 20)
    else:  # Post-monsoon autumn
        base_pm25 = np.random.normal(130, 18)
    
    # Diurnal (hourly) patterns
    if 5 <= hour <= 8:  # Morning rush + inversion
        diurnal_factor = 1.3
    elif 17 <= hour <= 21:  # Evening rush + cooking
        diurnal_factor = 1.2
    elif 12 <= hour <= 16:  # Midday mixing
        diurnal_factor = 0.8
    else:  # Night/early morning
        diurnal_factor = 1.0
    
    # Weather effects
    wind_effect = max(0.3, 1 - (wind_speed * 0.08))  # Wind reduces pollution
    humidity_effect = 1 + (humidity - 50) * 0.003    # High humidity traps pollution
    temp_effect = 1 - (temperature - 25) * 0.01      # Higher temp = more mixing
    
    # Apply all factors
    pm25 = base_pm25 * diurnal_factor * wind_effect * humidity_effect * temp_effect
    pm25 = max(pm25, 5)  # Never below 5 μg/m³
    
    # PM10 relationship with some randomness
    pm10_ratio = np.random.uniform(1.3, 1.7)  # Slightly wider range
    pm10 = pm25 * pm10_ratio
    
    return pd.Series([round(pm25, 1), round(pm10, 1)])

df[["pm25", "pm10"]] = df.apply(generate_pm25_pm10, axis=1)

# --------------------------
# AQI calculation (EPA) - Your original code is perfect
# --------------------------
pm25_breakpoints = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500)
]

pm10_breakpoints = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 604, 301, 500)
]

def pm_to_aqi(conc, breakpoints):
    for conc_lo, conc_hi, aqi_lo, aqi_hi in breakpoints:
        if conc_lo <= conc <= conc_hi:
            return round(((aqi_hi - aqi_lo) / (conc_hi - conc_lo)) * (conc - conc_lo) + aqi_lo)
    return 500  # beyond index

df["aqi_pm25"] = df["pm25"].apply(lambda x: pm_to_aqi(x, pm25_breakpoints))
df["aqi_pm10"] = df["pm10"].apply(lambda x: pm_to_aqi(x, pm10_breakpoints))

# Final AQI is the max of PM₂.₅ & PM₁₀ AQI
df["aqi"] = df[["aqi_pm25", "aqi_pm10"]].max(axis=1)

# --------------------------
# Add data quality indicators
# --------------------------
df["city"] = "Lahore"
df["country"] = "Pakistan"

# Optional: Add confidence score based on weather data completeness
df["data_quality"] = np.where(
    df[["temperature", "humidity", "pressure", "wind_speed"]].isnull().any(axis=1),
    "low", "high"
)

# --------------------------
# Save and validate
# --------------------------
output_df = df[["timestamp", "aqi", "pm25", "pm10", "city", "country"]]

# Data validation
assert output_df["aqi"].between(0, 500).all(), "AQI values out of valid range"
assert output_df["pm25"].min() >= 0, "Negative PM2.5 values found"
assert output_df["pm10"].min() >= 0, "Negative PM10 values found"

output_df.to_csv(aqi_file, index=False)

print(f"✅ Synthetic AQI data saved to {aqi_file}")
print(f"Total records: {len(output_df)}")
print(f"AQI range: {output_df['aqi'].min()} - {output_df['aqi'].max()}")
print(f"PM2.5 range: {output_df['pm25'].min():.1f} - {output_df['pm25'].max():.1f} μg/m³")
print(f"PM10 range: {output_df['pm10'].min():.1f} - {output_df['pm10'].max():.1f} μg/m³")