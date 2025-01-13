import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricePredictionModel:
    def __init__(self, processed_data_dir='data/processed', models_dir='models'):
        self.processed_data_dir = Path(processed_data_dir)
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
            # Load preprocessed data
            data_file = self.processed_data_dir / f'processed_data_{country_code}.csv'
            data = pd.read_csv(data_file, index_col=0, parse_dates=True)
            
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
            
            # Create sequences
            X_train, y_train = self.create_sequences(train_data_scaled, self.sequence_length)
            X_val, y_val = self.create_sequences(val_data_scaled, self.sequence_length)
            X_test, y_test = self.create_sequences(test_data_scaled, self.sequence_length)
            
            # Build and train model
            model = self.build_model(input_shape=(self.sequence_length, X_train.shape[2]))
            
            # Add early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Save model with .keras extension (new format)
            model_path = self.models_dir / f'lstm_model_{country_code}.keras'
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Also save the scalers for later use
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
    model_trainer = PricePredictionModel()
    
    # Train models for all available countries
    data_files = list(Path('data/processed').glob('processed_data_*.csv'))
    countries = [f.stem.split('processed_data_')[1] for f in data_files]
    
    for country_code in countries:
        try:
            model, history = model_trainer.train_model(country_code)
            logger.info(f"Successfully trained model for {country_code}")
        except Exception as e:
            logger.error(f"Error training model for {country_code}: {str(e)}")
            continue

if __name__ == "__main__":
    main()