import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricePredictor:
    def __init__(self, models_dir='models', data_dir='data', historical_data_dir='data/historical_data', results_dir='results'):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.historical_data_dir = Path(historical_data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = 24  # Must match training sequence length
        
    def load_model_and_scalers(self, country_code):
        """Load trained model and scalers for a country"""
        model_path = self.models_dir / f'lstm_model_{country_code}.keras'
        scaler_path = self.models_dir / f'scalers_{country_code}.pkl'
        
        model = tf.keras.models.load_model(model_path)
        scalers = joblib.load(scaler_path)
        
        return model, scalers['price_scaler'], scalers['feature_scaler']
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for prediction"""
        X = []
        for i in range(len(data) - sequence_length):
            X.append(data.iloc[i:(i + sequence_length)].values)
        return np.array(X)
    
    def evaluate_predictions(self, y_true, y_pred):
        """Calculate various error metrics"""
        # Remove any NaN values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
        
        # Calculate MAPE avoiding division by zero
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        metrics['MAPE'] = mape
        
        return metrics
    
    def plot_predictions(self, y_true, y_pred, dates, country_code, set_name='test'):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(15, 6))
        plt.plot(dates, y_true, label='Actual', alpha=0.8)
        plt.plot(dates, y_pred, label='Predicted', alpha=0.8)
        plt.title(f'Actual vs Predicted Prices - {country_code} ({set_name} set)')
        plt.xlabel('Date')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(self.results_dir / f'predictions_{country_code}_{set_name}.png')
        plt.close()
    
    def predict_and_evaluate(self, country_code):
        """Make predictions and evaluate model performance"""
        logger.info(f"Evaluating model for {country_code}")
        
        try:
            # Load model and scalers
            model, price_scaler, feature_scaler = self.load_model_and_scalers(country_code)
            
            # Load current and historical data
            data_frames = []
            
            # Load historical data if it exists
            historical_file = self.historical_data_dir / f'{country_code}.csv'
            if historical_file.exists():
                historical_data = pd.read_csv(historical_file, index_col=0, parse_dates=True)
                data_frames.append(historical_data)
                logger.info(f"Loaded historical data from {historical_file}")
            
            # Load current data if it exists
            current_file = self.data_dir / f'{country_code}.csv'
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
            
            # Split data (same as in training)
            train_size = int(len(data) * 0.7)
            val_size = int(len(data) * 0.15)
            test_data = data[train_size+val_size:]
            
            # Scale features
            feature_columns = [col for col in data.columns if col != 'price']
            test_price = test_data['price'].values.reshape(-1, 1)
            test_features = feature_scaler.transform(test_data[feature_columns])
            
            # Combine scaled data
            test_data_scaled = pd.DataFrame(
                np.hstack([price_scaler.transform(test_price), test_features]),
                index=test_data.index,
                columns=['price'] + feature_columns
            )
            
            # Verify no NaN values in scaled data
            if test_data_scaled.isnull().any().any():
                logger.error("Scaled data contains NaN values")
                raise ValueError("Scaling produced NaN values")
            
            # Create sequences
            X_test = self.create_sequences(test_data_scaled, self.sequence_length)
            
            # Make predictions
            y_pred_scaled = model.predict(X_test, verbose=0)  # Reduce verbosity
            
            # Inverse transform predictions
            y_pred = price_scaler.inverse_transform(y_pred_scaled)
            
            # Get actual values (excluding first sequence_length points)
            y_true = test_data['price'].values[self.sequence_length:]
            dates = test_data.index[self.sequence_length:]
            
            # Calculate metrics
            metrics = self.evaluate_predictions(y_true, y_pred.flatten())
            
            # Log metrics
            for metric_name, value in metrics.items():
                if np.isfinite(value):  # Only log finite values
                    logger.info(f"{metric_name}: {value:.4f}")
                else:
                    logger.warning(f"{metric_name}: Invalid value")
            
            # Save metrics
            pd.DataFrame(metrics, index=[0]).to_csv(
                self.results_dir / f'metrics_{country_code}.csv'
            )
            
            # Save predictions and actual values
            predictions_df = pd.DataFrame({
                'timestamp': dates,
                'actual': y_true,
                'predicted': y_pred.flatten()
            })
            predictions_df.set_index('timestamp', inplace=True)
            predictions_df.to_csv(self.results_dir / f'predictions_{country_code}.csv')
            logger.info(f"Saved predictions to {self.results_dir}/predictions_{country_code}.csv")
            
            # Plot results
            self.plot_predictions(y_true, y_pred.flatten(), dates, country_code)
            
            return metrics, y_true, y_pred, dates
            
        except Exception as e:
            logger.error(f"Error evaluating model for {country_code}: {str(e)}")
            raise

def main():
    # Get project root directory (2 levels up from this script)
    project_root = Path(__file__).parents[2]
    
    predictor = PricePredictor(
        models_dir=project_root / 'models',
        data_dir=project_root / 'data',
        historical_data_dir=project_root / 'data/historical_data',
        results_dir=project_root / 'results'
    )
    
    # Read available bidding zones from CSV
    bidding_zones_df = pd.read_csv(project_root / 'data' / 'available_bidding_zones.csv')
    # Filter for active zones only
    active_zones = bidding_zones_df[bidding_zones_df['active'] == 1]
    logger.info(f"Found {len(active_zones)} active bidding zones")
    
    # Get list of trained models
    existing_models = set(f.stem.replace('lstm_model_', '') 
                         for f in (project_root / 'models').glob('lstm_model_*.keras'))
    logger.info(f"Found trained models for: {', '.join(sorted(existing_models))}")
    
    results = {}
    for _, row in active_zones.iterrows():
        country_code = row['short_code']  # Use short code for file names
        
        # Skip if no model exists
        if country_code not in existing_models:
            logger.warning(f"No trained model found for {country_code}, skipping evaluation")
            continue
            
        try:
            metrics, _, _, _ = predictor.predict_and_evaluate(country_code)
            results[country_code] = metrics
        except Exception as e:
            logger.error(f"Failed to evaluate {country_code}: {str(e)}")
            continue
    
    # Print summary of results
    print("\nSummary of Results:")
    for country, metrics in results.items():
        print(f"\n{country}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()