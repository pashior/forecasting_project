# Indoor Risk Forecasting API

A machine learning-powered API for predicting indoor environmental risk levels using outdoor weather data and limited indoor measurements. The system leverages temporal patterns and external environmental conditions to forecast indoor risk factors when direct indoor monitoring is unavailable or insufficient.

## 🎯 Overview

This project addresses the challenge of predicting indoor environmental risk (such as air quality, health hazards, or comfort levels) in scenarios where comprehensive indoor sensor data is limited. By combining outdoor meteorological and air quality data with sparse indoor measurements, the system learns complex mappings between external conditions, temporal patterns, and indoor risk factors.

## 🔬 Scientific Background

Indoor environments are significantly influenced by:
- **Outdoor weather conditions** (temperature, humidity, pressure, wind)
- **Air quality parameters** (PM10, CO, CO₂, NO₂)
- **Temporal patterns** reflecting human activity cycles
- **Building characteristics** and occupancy patterns

This system employs a novel neural network architecture that guarantees the inclusion of critical temporal features while learning from limited indoor data to predict future risk levels.

## 🏗️ System Architecture

### Data Integration Pipeline

```
Outdoor Data (Open-Meteo) + Indoor Data (CSV) → Feature Engineering → Model Training → Risk Prediction
```

### Model Architecture: Guaranteed Feature Model

The core of the system is a custom TensorFlow model (`GuaranteedFeatureModel`) with:

- **Dual-Network Architecture**:
  - **Base Network**: Processes all environmental and temporal features
  - **Critical Network**: Focuses specifically on temporal patterns (guaranteed features)
  - **Adaptive Fusion**: Dynamically combines outputs using learned weighting

- **Temporal Feature Engineering**:
  - Cyclic encoding of time components (hour, day, month, day-of-week)
  - Captures seasonal and daily human activity patterns
  - Ensures temporal awareness in predictions

- **Regularization Strategy**:
  - L2 regularization and dropout for generalization
  - Handles limited training data effectively

## 📊 Data Sources and Features

### External Data (Open-Meteo APIs)
- **Weather Parameters**: Temperature, humidity, dew point, pressure, wind speed
- **Air Quality**: PM10, Carbon monoxide, Carbon dioxide, Nitrogen dioxide
- **Temporal Coverage**: Historical and forecast data

### Internal Data (User-Provided)
- **Format**: CSV with timestamp and risk columns
- **Content**: Limited indoor risk measurements
- **Purpose**: Ground truth for model training

### Engineered Features
- **Temporal**: Sinusoidal encoding of cyclical time components
- **Environmental**: Standardized outdoor measurements
- **Spatial**: Location-specific data (latitude, longitude)

## 🚀 API Endpoints

### `POST /register`
**Register a new user and train a personalized model**

**Parameters:**
- `file`: CSV file with indoor risk data (columns: `time`, `risk`)
- `unique_id`: User identifier
- `latitude`: Location latitude
- `longitude`: Location longitude

**Response:**
```json
{
  "rows": 1000,
  "unique_id": "user123",
  "latitude": 41.0082,
  "longitude": 28.9784,
  "data": [...],
  "alpha": 0.65
}
```

### `GET /predict`
**Generate risk predictions for a time period**

**Parameters:**
- `unique_id`: User identifier
- `start_date`: Prediction start date (YYYY-MM-DD)
- `end_date`: Prediction end date (YYYY-MM-DD)

**Response:**
```json
{
  "latitude": 41.0082,
  "longitude": 28.9784,
  "df_length": 168,
  "predictions": [[34.5], [38.7], ...]
}
```

### `PUT /update`
**Update model with new indoor data**

**Parameters:**
- `unique_id`: User identifier
- `file`: Updated CSV file
- `latitude`: Optional updated latitude
- `longitude`: Optional updated longitude

### `POST /delete`
**Remove user data and model**

**Parameters:**
- `unique_id`: User identifier

## 📈 Methodology

### 1. Data Preprocessing
```python
# Temporal feature engineering
df['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
# ... similar for month, day_of_week, day_of_month

# Standardization of environmental features
scaler = StandardScaler()
df[environmental_features] = scaler.fit_transform(df[environmental_features])
```

### 2. Model Training
- **Architecture**: Guaranteed Feature Model with dual networks
- **Optimization**: Adam optimizer with early stopping
- **Validation**: Train-test split (75-25)
- **Metrics**: MSE loss, MAE for evaluation

### 3. Prediction Pipeline
- Fetch outdoor forecast data for specified period
- Apply learned feature transformations
- Generate risk predictions using trained model
- Return timestamped risk forecasts

## 🛠️ Installation and Setup

### Prerequisites
```bash
Python >= 3.8
TensorFlow >= 2.x
FastAPI
scikit-learn
pandas
joblib
requests
```

### Installation
```bash
# Clone repository
git clone https://github.com/pashior/forecasting_project.git
cd forecasting_project

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📝 Usage Example

### 1. Prepare Indoor Data
```csv
time,risk
2024-01-01 00:00:00,0.3
2024-01-01 01:00:00,0.25
2024-01-01 02:00:00,0.2
...
```

### 2. Register User and Train Model
```python
import requests

files = {'file': open('indoor_data.csv', 'rb')}
data = {
    'unique_id': 'building_A',
    'latitude': 41.0082,
    'longitude': 28.9784
}

response = requests.post('http://localhost:8000/register', files=files, data=data)
```

### 3. Generate Predictions
```python
params = {
    'unique_id': 'building_A',
    'start_date': '2024-02-01',
    'end_date': '2024-02-07'
}

response = requests.get('http://localhost:8000/predict', params=params)
predictions = response.json()['predictions']
```

## 🔬 Technical Details

### Model Architecture
```python
class GuaranteedFeatureModel(tf.keras.Model):
    def __init__(self, d_all, critical_idx, alpha_min=0.4, ...):
        # Dual network architecture
        self.base_net = tf.keras.Sequential([...])      # All features
        self.critical_net = tf.keras.Sequential([...])  # Temporal features
        self.alpha = learnable_weight()                 # Fusion parameter
    
    def call(self, x):
        base_out = self.base_net(x)
        critical_out = self.critical_net(x[critical_features])
        return base_out + self.alpha * critical_out
```

### Feature Importance
- **Temporal Features**: Guaranteed inclusion via critical network
- **Environmental Features**: Processed through base network
- **Adaptive Weighting**: Model learns optimal feature combination

## 📊 Performance Considerations

### Model Generalization
- **Regularization**: L2 penalties and dropout layers
- **Early Stopping**: Prevents overfitting on limited data
- **Cross-Validation**: Robust model selection

### Scalability
- **User Isolation**: Individual models per user/location
- **Efficient Storage**: Compressed model artifacts
- **API Performance**: FastAPI for high-throughput serving

## 🔮 Future Enhancements

1. **Multi-Location Models**: Transfer learning across locations
2. **Real-Time Updates**: Streaming data integration
3. **Uncertainty Quantification**: Probabilistic predictions
4. **Feature Selection**: Automated critical feature identification
5. **Model Ensemble**: Combining multiple prediction strategies

## 📚 Scientific Applications

- **Building Management**: HVAC optimization and energy efficiency
- **Public Health**: Indoor air quality monitoring and alerts
- **Smart Cities**: Urban environmental risk assessment
- **Research**: Indoor-outdoor environmental correlation studies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Maintainer**: [@pashior](https://github.com/pashior)

For questions, bug reports, or collaboration opportunities, please:
- Open an issue on GitHub
- Email: brt.klc795@gmail.com
- Join our discussions in the repository

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{forecasting_project,
  title={Indoor Risk Forecasting API: Predicting Indoor Environmental Risk Using Outdoor Data and Temporal Patterns},
  author={[Your Name]},
  year={2025},
  url={https://github.com/pashior/forecasting_project}
}
```

---

*This project demonstrates the application of machine learning to environmental monitoring, bridging the gap between outdoor observations and indoor risk assessment through innovative neural network architectures and temporal pattern recognition.*
