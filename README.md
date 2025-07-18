# Infection Risk Forecasting with LSTM Neural Network

This project implements a time-series forecasting model using a Long Short-Term Memory (LSTM) neural network to predict `InfectionRisk`. The model is built with TensorFlow and Keras.

## 📋 Project Overview

The primary goal of this project is to predict the `InfectionRisk` for an upcoming week based on data from the previous three weeks. The time-series data is processed, scaled, and then fed into a stacked LSTM model for training and prediction.

## 📂 Dataset

* **Source:** The model is trained on a time-series dataset loaded from a CSV file located at `4.08-7.03.csv`.
* **Target Variable:** `InfectionRisk`
* **Features:** All other columns in the dataset after dropping identifiers and metadata.
* **Preprocessing:**
    * Features and the target variable are scaled using `StandardScaler` from scikit-learn.
    * The data is structured into windows, where each sample consists of 3 weeks of data (input) to predict the following week (output).

## 🛠️ Model Architecture

The forecasting model is a `Sequential` neural network built with TensorFlow/Keras, consisting of the following layers:

1.  **Input Layer:** Configured for the shape of the training data.
2.  **LSTM Layer 1:** 159 units, returns sequences for the next LSTM layer.
3.  **LSTM Layer 2:** 172 units.
4.  **Dense Layer:** 123 units with a `tanh` activation function.
5.  **Output Layer:** A `Dense` layer with 168 units (for 168 hourly predictions in a week) and a `linear` activation function.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. The required libraries are listed in the `requirements.txt` file.

### Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/pashior/forecasting_project.git
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### `requirements.txt`

```
pandas
numpy
tensorflow
scikit-learn
matplotlib
random
```

### Usage

1.  **Data:** Place your dataset CSV file in the correct path or update the path in the notebook.
2.  **Training:** Run the Jupyter Notebook cells sequentially to preprocess the data, build the model, and initiate training.
3.  **Model Checkpoint:** The best-performing model (based on `val_pearson_correlation`) is automatically saved as `model.keras`.

## 📈 Results
- Weights & Biases platform was used to log model metrics.  
- You can see the results for 10 different models [here](https://api.wandb.ai/links/brtklc795-marmara-niversitesi/mikmcf0y).   
- The best performing model was trained for 69 epochs, achieving the following performance on the validation set:  
-**Validation Loss (MSE):** ~0.0081  
-**Validation MAE:** ~0.0590  
-**Validation Pearson Correlation:** ~0.9820

### *You will find deatiled plots and explanations in the notebook file.*

## ✍️ Author

* Berat Kilic

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
