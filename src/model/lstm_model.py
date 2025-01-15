import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricePredictionModel:
    def __init__(self, data_dir='data', historical_data_dir='data/historical_data', models_dir='models'):
        self.data_dir = Path(data_dir)
        self.historical_data_dir = Path(historical_data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.sequence_length = 24  # Use last 24 hours to predict next hour
        self.batch_size = 32
        self.epochs = 50
        
        # Initialize scalers
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        
    def create_sequences(self, data, sequence_length):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data.iloc[i:(i + sequence_length)].values)
            y.append(data.iloc[i + sequence_length]['price'])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Define LSTM model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, country_code='DE_LU'):
        """Train model for a specific country"""
        logger.info(f"Training model for {country_code}")
        
        try:
            # Load current and historical data
            current_file = self.data_dir / f'{country_code}.csv'
            historical_file = self.historical_data_dir / f'{country_code}.csv'
            
            data_frames = []
            
            # Load historical data if it exists
            if historical_file.exists():
                historical_data = pd.read_csv(historical_file, index_col=0, parse_dates=True)
                data_frames.append(historical_data)
                logger.info(f"Loaded historical data from {historical_file}")
            
            # Load current data if it exists
            if current_file.exists():
                current_data = pd.read_csv(current_file, index_col=0, parse_dates=True)
                data_frames.append(current_data)
                logger.info(f"Loaded current data from {current_file}")
            
            if not data_frames:
                raise FileNotFoundError(f"No data found for {country_code}")
            
            # Combine and sort data
            data = pd.concat(data_frames)
            data = data.sort_index().drop_duplicates()
            
            # Handle missing or infinite values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.dropna()
            logger.info(f"Total data points after cleaning: {len(data)}")
            
            # Check for remaining issues
            if data.isnull().any().any():
                logger.error("Data still contains NaN values after cleaning")
                raise ValueError("Data contains NaN values")
            
            if (data['price'] <= 0).any():
                logger.warning("Data contains non-positive prices, adding offset")
                min_price = data['price'].min()
                if min_price <= 0:
                    data['price'] = data['price'] - min_price + 1  # Ensure all prices are positive
            
            # Split data into train/val/test (70/15/15)
            train_size = int(len(data) * 0.7)
            val_size = int(len(data) * 0.15)
            
            train_data = data[:train_size]
            val_data = data[train_size:train_size+val_size]
            test_data = data[train_size+val_size:]
            
            # Scale the data
            feature_columns = [col for col in data.columns if col != 'price']
            
            # Fit scalers on training data only
            self.price_scaler.fit(train_data[['price']])
            self.feature_scaler.fit(train_data[feature_columns])
            
            # Transform all data
            train_data_scaled = self.transform_data(train_data)
            val_data_scaled = self.transform_data(val_data)
            test_data_scaled = self.transform_data(test_data)
            
            # Verify scaled data
            if train_data_scaled.isnull().any().any():
                logger.error("Scaled data contains NaN values")
                raise ValueError("Scaling produced NaN values")
            
            # Create sequences
            X_train, y_train = self.create_sequences(train_data_scaled, self.sequence_length)
            X_val, y_val = self.create_sequences(val_data_scaled, self.sequence_length)
            X_test, y_test = self.create_sequences(test_data_scaled, self.sequence_length)
            
            # Build model with Input layer explicitly defined
            input_shape = (self.sequence_length, X_train.shape[2])
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            
            # Use gradient clipping to prevent exploding gradients
            optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
            
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            # Add early stopping and reduce learning rate on plateau
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save model and scalers
            model_path = self.models_dir / f'lstm_model_{country_code}.keras'
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            scaler_path = self.models_dir / f'scalers_{country_code}.pkl'
            import joblib
            joblib.dump({
                'price_scaler': self.price_scaler,
                'feature_scaler': self.feature_scaler
            }, scaler_path)
            logger.info(f"Scalers saved to {scaler_path}")
            
            # Evaluate on test set
            test_loss = model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"Test loss: {test_loss}")
            
            return model, history
            
        except Exception as e:
            logger.error(f"Error training model for {country_code}: {str(e)}")
            raise
    
    def transform_data(self, data):
        """Scale the features and price"""
        feature_columns = [col for col in data.columns if col != 'price']
        
        # Scale price and features separately
        price_scaled = self.price_scaler.transform(data[['price']])
        features_scaled = self.feature_scaler.transform(data[feature_columns])
        
        # Combine scaled data
        scaled_data = pd.DataFrame(
            np.hstack([price_scaled, features_scaled]),
            index=data.index,
            columns=['price'] + feature_columns
        )
        
        return scaled_data

def main():
    # Get project root directory (2 levels up from this script)
    project_root = Path(__file__).parents[2]
    
    model_trainer = PricePredictionModel(
        data_dir=project_root / 'data',
        historical_data_dir=project_root / 'data/historical_data',
        models_dir=project_root / 'models'
    )
    
    # Read available bidding zones from CSV
    bidding_zones_df = pd.read_csv(project_root / 'data' / 'available_bidding_zones.csv')
    # Filter for active zones only
    active_zones = bidding_zones_df[bidding_zones_df['active'] == 1]
    logger.info(f"Found {len(active_zones)} active bidding zones")
    
    # Get list of existing models
    existing_models = set(f.stem.replace('lstm_model_', '') 
                         for f in (project_root / 'models').glob('lstm_model_*.keras'))
    logger.info(f"Found existing models for: {', '.join(sorted(existing_models))}")
    
    # Train models for all active zones
    for _, row in active_zones.iterrows():
        country_code = row['short_code']  # Use short code for file names
        try:
            # Check if model already exists
            if country_code in existing_models:
                logger.info(f"Model already exists for {country_code}, skipping training")
                continue
                
            model, history = model_trainer.train_model(country_code)
            logger.info(f"Successfully trained model for {country_code}")
        except Exception as e:
            logger.error(f"Error training model for {country_code}: {str(e)}")
            continue

if __name__ == "__main__":
    main()