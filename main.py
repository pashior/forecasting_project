from sklearn.preprocessing import StandardScaler   
from sklearn.model_selection import train_test_split 
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
import pandas as pd
import os
import requests
import joblib
import json
import shutil

app = FastAPI()

@tf.keras.utils.register_keras_serializable()
class GuaranteedFeatureModel(tf.keras.Model):
    def __init__(
        self,
        d_all,
        critical_idx,
        alpha_min=0.1,
        hidden_base=32,
        hidden_critical=16,
        l2_base=0.01,
        l2_critical=0.005,
        trainable=True,
        dtype=None,
        **kwargs
    ):
        super().__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.d_all = d_all
        self.critical_idx = critical_idx if isinstance(critical_idx, list) else list(critical_idx)
        self.alpha_min = alpha_min
        self.hidden_base = hidden_base
        self.hidden_critical = hidden_critical
        self.l2_base = l2_base
        self.l2_critical = l2_critical

        # Index tensor
        self.critical_idx_tensor = tf.constant(self.critical_idx, dtype=tf.int32)

        # Networks
        self.base_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_base, activation='softplus',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_base)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(hidden_base // 2, activation='softplus',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_base)),
            tf.keras.layers.Dense(1)
        ])
        self.critical_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_critical, activation='softplus',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_critical)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])

        # Alpha parameter (learnable)
        self.s = self.add_weight(name='alpha_logit', shape=(), initializer='zeros', trainable=True)

    @property
    def alpha(self):
        # alpha in [alpha_min, 1]
        return self.alpha_min + (1.0 - self.alpha_min) * tf.sigmoid(self.s)

    def call(self, x, training=False):
        x_critical = tf.gather(x, self.critical_idx_tensor, axis=1)

        # Two branches
        base_out = self.base_net(x, training=training)
        critical_out = self.critical_net(x_critical, training=training)

        # Merge and positive output
        total_out = base_out + self.alpha * critical_out
        return tf.nn.softplus(total_out)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_all': self.d_all,
            'critical_idx': self.critical_idx,
            'alpha_min': self.alpha_min,
            'hidden_base': self.hidden_base,
            'hidden_critical': self.hidden_critical,
            'l2_base': self.l2_base,
            'l2_critical': self.l2_critical,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def get_air_quality_data(latitude: float, longitude: float, start_date: str=None, end_date: str=None):
    timezone = "auto"
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(["pm10", "carbon_monoxide", "carbon_dioxide",
                            "nitrogen_dioxide"
                            ]),
        "timezone": timezone
    }
    response = requests.get(url, params=params)
    data = response.json()
    df=pd.DataFrame(data["hourly"])
    df["time"]=pd.to_datetime(df["time"])
    return df

def get_weather_data(latitude: float, longitude: float, start_date: str=None, end_date: str=None):
    timezone = "auto"
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(['temperature_2m', "relative_humidity_2m",
                            "dew_point_2m",'pressure_msl', 'wind_speed_10m'
                            ]),
        "timezone": timezone
    }
    response = requests.get(url, params=params)
    data = response.json()
    df=pd.DataFrame(data["hourly"])
    df["time"]=pd.to_datetime(df["time"])
    return df

def merge_data(risk_data: pd.DataFrame, latitude: float, longitude: float):
    risk_data = risk_data[["time","risk"]]
    risk_data["time"] = pd.to_datetime(risk_data["time"])
    start_date = risk_data["time"].min().strftime("%Y-%m-%d")
    end_date = risk_data["time"].max().strftime("%Y-%m-%d")

    past_weather = get_weather_data(latitude, longitude, start_date=start_date, end_date=end_date)
    air_quality = get_air_quality_data(latitude, longitude, start_date=start_date, end_date=end_date)
    # Merge both DataFrames respect to time
    merged_df = pd.merge(past_weather,
                               air_quality,
                               on="time", how="inner")
    df = pd.merge(merged_df, risk_data, on="time", how="inner")

    df['month_sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['time'].dt.month / 12)

    df['day_of_month_sin'] = np.sin(2 * np.pi * df['time'].dt.day / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['time'].dt.day / 31)

    df['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)

    df['day_of_week_sin'] = np.sin(2 * np.pi * df['time'].dt.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['time'].dt.dayofweek / 7)

    return df

def get_forecast_data(latitude: float, longitude: float, start_date: str=None, end_date: str=None):
    weather = get_weather_data(latitude, longitude, start_date=start_date, end_date=end_date)
    air_quality = get_air_quality_data(latitude, longitude, start_date=start_date, end_date=end_date)

    # Merge both DataFrames respect to time
    df = pd.merge(weather, air_quality, on="time", how="inner")

    df['month_sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['time'].dt.month / 12)

    df['day_of_month_sin'] = np.sin(2 * np.pi * df['time'].dt.day / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['time'].dt.day / 31)

    df['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)

    df['day_of_week_sin'] = np.sin(2 * np.pi * df['time'].dt.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['time'].dt.dayofweek / 7)

    return df

def preprocess_data(df: pd.DataFrame):
    # Normalize features
    scaler = StandardScaler()
    scale_columns = [
              'temperature_2m', "relative_humidity_2m",
              "dew_point_2m",'pressure_msl', 'wind_speed_10m',
              "pm10", "carbon_monoxide",
              "nitrogen_dioxide", "carbon_dioxide"
              ]
    feature_columns = [
              'hour_sin', 'hour_cos',
              "day_of_week_sin","day_of_week_cos",
              'day_of_month_sin', 'day_of_month_cos',
              'month_sin', 'month_cos',
              'temperature_2m', "relative_humidity_2m",
              "dew_point_2m",'pressure_msl', 'wind_speed_10m',
              "pm10", "carbon_monoxide",
              "nitrogen_dioxide", "carbon_dioxide"
              ]
    df[scale_columns] = scaler.fit_transform(df[scale_columns])

    X = df[feature_columns].to_numpy()
    y = df['risk'].to_numpy().reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return x_train, x_test, y_train, y_test, feature_columns, scale_columns, scaler

def build_model(d_all, critical_features=[0, 1, 2, 3, 4, 5], alpha_min=0.4, l2_base=0.01, l2_critical=0.005):
    model = GuaranteedFeatureModel(
        d_all=d_all,
        critical_idx=critical_features,
        alpha_min=alpha_min,
        l2_base=l2_base,
        l2_critical=l2_critical
    )
    return model

def train_and_select_best_model(x_train, y_train, x_test, y_test, learning_rate=0.01, epochs=200, batch_size=64):

    tf.config.run_functions_eagerly(True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for i in range(1):
        model = build_model(d_all=x_train.shape[1])
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=[early_stopping])

        if i == 0:
            best_model = model
            best_history = history
        else:
            if history.history['val_loss'][-1] < best_history.history['val_loss'][-1]:
                best_model = model
                best_history = history

    return best_model

@app.post("/register")
async def register(file: UploadFile = File(...),
                   unique_id: str = Form(...),
                   latitude: float = Form(...),
                   longitude: float = Form(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="uploaded file is not in csv format!")

    try:
        df = pd.read_csv(file.file, sep=None, engine="python")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot read file!: {e}")


    df = merge_data(df, latitude, longitude)
    x_train, x_test, y_train, y_test, feature_columns, scale_columns, scaler = preprocess_data(df)
    model = train_and_select_best_model(x_train, y_train, x_test, y_test)

    # Create user directory
    user_dir = os.path.join("users", unique_id)
    os.makedirs(user_dir, exist_ok=True)

    # Save original CSV
    orig_csv_path = os.path.join(user_dir, "original.csv")
    df.to_csv(orig_csv_path, index=False)

    # Save scaler
    scaler_path = os.path.join(user_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Save feature order
    features_path = os.path.join(user_dir, "features.txt")
    with open(features_path, "w") as f:
        f.write("\n".join(feature_columns))

    # Save scaled columns
    scale_columns_path = os.path.join(user_dir, "scale_columns.txt")
    with open(scale_columns_path, "w") as f:
        f.write("\n".join(scale_columns))

    # Save coordinates
    coordinates_path = os.path.join(user_dir, "coordinates.json")
    with open(coordinates_path, "w") as f:
        json.dump({"latitude": latitude, "longitude": longitude}, f)

    # Save model
    model_path = os.path.join(user_dir, "risk_model.keras")
    model.save(model_path)

    return {
        "rows": int(len(df)),
        "unique_id": unique_id,
        "latitude": latitude,
        "longitude": longitude,
        "data": df.head(10).to_dict(orient="records"),
        "alpha": model.alpha.numpy().tolist(),
    }

@app.get("/predict")
async def predict(unique_id: str, start_date: str = None, end_date: str = None):
    user_dir = os.path.join("users", unique_id)
    model_path = os.path.join(user_dir, "risk_model.keras")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    model = tf.keras.models.load_model(model_path)

    # Scale the data
    scaler_path = os.path.join(user_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    coordinates_path = os.path.join(user_dir, "coordinates.json")
    with open(coordinates_path, "r") as f:
        coordinates = json.load(f)
    latitude, longitude = coordinates.get("latitude"), coordinates.get("longitude")

    df = get_forecast_data(latitude, longitude, start_date=start_date, end_date=end_date)

    scale_columns_path = os.path.join(user_dir, "scale_columns.txt")
    with open(scale_columns_path, "r") as f:
        scale_columns = f.read().splitlines()

    feature_path = os.path.join(user_dir, "features.txt")
    with open(feature_path, "r") as f:
        feature_columns = f.read().splitlines()

    df[scale_columns] = scaler.transform(df[scale_columns])
    data = df[feature_columns].to_numpy()

    # Make predictions
    predictions = model.predict(data)
    pd.DataFrame(predictions, columns=["risk"]).to_csv(os.path.join(user_dir, "predictions.csv"), index=False)
    return {"f": feature_columns,
            "df": df.to_dict(orient="records"),
            "latitude": latitude,
            "longitude": longitude,
            "df_length": len(df),
            "predictions": predictions.tolist()
    }

@app.put("/update")
async def update(unique_id: str, latitude: float=None, longitude: float=None, file: UploadFile = File(...)):
    user_dir = os.path.join("users", unique_id)
    if not os.path.exists(user_dir):
        raise HTTPException(status_code=404, detail="User not found")

    # Read the new data
    try:
        df = pd.read_csv(file.file, sep=None, engine="python")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot read file!: {e}")

    # Get coordinates
    if latitude is None or longitude is None:
        coordinates_path = os.path.join(user_dir, "coordinates.json")
        with open(coordinates_path, "r") as f:
            coordinates = json.load(f)
        latitude, longitude = coordinates.get("latitude"), coordinates.get("longitude")

    df = merge_data(df, latitude, longitude)
    x_train, x_test, y_train, y_test, feature_columns, scale_columns, scaler = preprocess_data(df)
    model = train_and_select_best_model(x_train, y_train, x_test, y_test)

    # Save original CSV
    orig_csv_path = os.path.join(user_dir, "original.csv")
    df.to_csv(orig_csv_path, index=False)

    # Save scaler
    scaler_path = os.path.join(user_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Save feature order
    features_path = os.path.join(user_dir, "features.txt")
    with open(features_path, "w") as f:
        f.write("\n".join(feature_columns))

    # Save scaled columns
    scale_columns_path = os.path.join(user_dir, "scale_columns.txt")
    with open(scale_columns_path, "w") as f:
        f.write("\n".join(scale_columns))

    # Save coordinates
    coordinates_path = os.path.join(user_dir, "coordinates.json")
    with open(coordinates_path, "w") as f:
        json.dump({"latitude": latitude, "longitude": longitude}, f)

    # Save model
    model_path = os.path.join(user_dir, "risk_model.keras")
    model.save(model_path)

    return {
        "rows": int(len(df)),
        "unique_id": unique_id,
        "latitude": latitude,
        "longitude": longitude,
        "data": df.head(10).to_dict(orient="records"),
        "alpha": model.alpha.numpy().tolist(),
    }

@app.post("/delete")
async def delete(unique_id: str):
    user_path=os.path.join("users", unique_id)
    if os.path.exists(user_path):
        shutil.rmtree(user_path)
        return {"detail": "User data deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="User not found")