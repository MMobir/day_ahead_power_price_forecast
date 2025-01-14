# Day-Ahead Power Price Forecasting

LSTM-based forecasting model for European electricity day-ahead prices.

## Overview

This project predicts day-ahead electricity prices for various European bidding zones using LSTM neural networks. The model considers multiple features including:
- Historical prices
- Load forecasts
- Wind and solar generation forecasts
- Time-based features

## Project Structure
```
├── data/
│   ├── raw/                    # Raw data from ENTSO-E API
│   └── processed/              # Preprocessed data ready for training
├── models/                     # Trained models and scalers
├── results/                    # Prediction plots and metrics
└── src/
    ├── data_preparation/
    │   ├── fetch_historical_data.py  # Fetches data from ENTSO-E
    │   └── preprocess_data.py        # Prepares data for training
    └── model/
        ├── lstm_model.py       # LSTM model definition and training
        └── predict.py          # Model evaluation and predictions
```

## Features

- Data fetching from ENTSO-E API for multiple European bidding zones
- Comprehensive data preprocessing pipeline
- LSTM-based deep learning model with:
  - Dual LSTM layers (64 and 32 units)
  - Dropout layers for regularization
  - Time-based feature engineering
  - Sequence-based prediction (24-hour lookback)
- Model evaluation with multiple metrics (MAE, RMSE, MAPE, R²)
- Visualization of predictions

## Requirements

- Python 3.8+
- Dependencies:
  ```
  numpy>=1.21.0
  pandas>=1.3.0
  tensorflow>=2.10.0
  scikit-learn>=1.0.0
  matplotlib>=3.4.0
  python-dotenv>=0.19.0
  joblib>=1.1.0
  requests>=2.26.0
  ```

## Setup

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd forecast_energy_price
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your ENTSO-E API key:
   ```
   ENTSOE_API_KEY=your_api_key_here
   ```

## Usage

1. Fetch historical data:
   ```bash
   python src/data_preparation/fetch_historical_data.py
   ```

2. Preprocess the data:
   ```bash
   python src/data_preparation/preprocess_data.py
   ```

3. Train the model:
   ```bash
   python src/model/lstm_model.py
   ```

4. Make predictions and evaluate:
   ```bash
   python src/model/predict.py
   ```

## Model Architecture

The LSTM model architecture consists of:
- Input layer with 24-hour sequence length
- First LSTM layer (64 units) with return sequences
- Dropout layer (20%)
- Second LSTM layer (32 units)
- Dropout layer (20%)
- Dense layer (16 units) with ReLU activation
- Output Dense layer (1 unit)

## Data Features

The model uses the following features:
- Historical prices (24h, 48h, and 168h lags)
- Load forecasts
- Wind generation forecasts (onshore/offshore)
- Solar generation forecasts
- Time-based features (hour, day of week, month, weekend)
- Cyclical encodings for time features
- Rolling statistics (24h mean and standard deviation)

## Results

Results are saved in the `results/` directory:
- Prediction plots for each country
- CSV files with evaluation metrics
- Performance metrics include MAE, RMSE, MAPE, and R²

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here] 