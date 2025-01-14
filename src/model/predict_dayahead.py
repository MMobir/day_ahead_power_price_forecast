import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import joblib
import logging
from datetime import datetime, timedelta
import pytz
from entsoe import EntsoePandasClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DayAheadPredictor:
    def __init__(self, models_dir='models', processed_data_dir='data/processed', results_dir='results'):
        self.models_dir = Path(models_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.results_dir = Path(results_dir)
        self.dayahead_dir = self.results_dir / 'dayahead'
        self.dayahead_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = 24  # Must match training sequence length
        self.client = EntsoePandasClient(api_key=os.getenv('ENTSOE_API_KEY'))
    
    def check_prediction_time(self):
        """Check optimal time to make predictions"""
        brussels_tz = pytz.timezone('Europe/Brussels')
        now = pd.Timestamp.now(tz=brussels_tz)
        
        # Evening run (main prediction for next day)
        # Run after 18:15 to ensure we have all ENTSOE forecasts
        if now.hour == 18 and now.minute >= 15:
            logger.info("Running evening predictions for next day")
            return "evening"
        
        # Morning run (update with latest forecasts)
        # Run at 6:00 for intraday insights
        if now.hour == 6 and now.minute <= 15:
            logger.info("Running morning update for current day")
            return "morning"
        
        logger.info("Not optimal prediction time. Best times: 18:15 CET (main) and 06:00 CET (update)")
        return None
    
    def fetch_tomorrow_forecasts(self, country_code):
        """Fetch tomorrow's load and generation forecasts from ENTSOE"""
        brussels_tz = pytz.timezone('Europe/Brussels')
        now = pd.Timestamp.now(tz=brussels_tz)
        tomorrow = now.date() + timedelta(days=1)
        
        # Time range for tomorrow 00:00 to 23:59
        start = pd.Timestamp.combine(tomorrow, pd.Timestamp.min.time()).tz_localize(brussels_tz)
        end = start + pd.Timedelta(days=1)
        
        logger.info(f"Fetching forecasts for {tomorrow}")
        
        try:
            # Fetch load forecast
            load_forecast = self.client.query_load_forecast(
                country_code,
                start=start,
                end=end,
                process_type="A01"
            )
            
            # Fetch wind and solar forecast
            generation_forecast = self.client.query_wind_and_solar_forecast(
                country_code,
                start=start,
                end=end
            )
            
            if load_forecast.empty or generation_forecast.empty:
                raise ValueError("No forecasts available yet for tomorrow")
            
            # Process generation forecast
            wind_offshore = generation_forecast.get('Wind Offshore', pd.Series(0, index=load_forecast.index))
            wind_onshore = generation_forecast.get('Wind Onshore', generation_forecast.get('Wind', pd.Series(0, index=load_forecast.index)))
            solar = generation_forecast.get('Solar', pd.Series(0, index=load_forecast.index))
            
            # Combine forecasts
            forecasts = pd.DataFrame({
                'load_forecast': load_forecast,
                'wind_offshore': wind_offshore,
                'wind_onshore': wind_onshore,
                'solar': solar
            })
            
            return forecasts, tomorrow
            
        except Exception as e:
            logger.error(f"Error fetching forecasts: {str(e)}")
            raise
    
    def prepare_features(self, forecasts):
        """Prepare features in the same way as training data"""
        features = forecasts.copy()
        
        # Calculate derived features
        features['total_wind_forecast'] = features['wind_offshore'] + features['wind_onshore']
        features['renewable_ratio'] = (features['total_wind_forecast'] + features['solar']) / features['load_forecast']
        
        # Add time-based features
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['month'] = features.index.month
        features['weekend'] = features.index.dayofweek.isin([5, 6]).astype(int)
        
        # Add cyclical features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour']/24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour']/24)
        features['month_sin'] = np.sin(2 * np.pi * features['month']/12)
        features['month_cos'] = np.cos(2 * np.pi * features['month']/12)
        
        return features
    
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
        for i in range(len(data) - sequence_length + 1):
            X.append(data.iloc[i:(i + sequence_length)].values)
        return np.array(X)
    
    def predict_next_day(self, country_code):
        """Generate predictions for the next 24 hours"""
        logger.info(f"Generating predictions for {country_code}")
        
        # Check if it's prediction time
        run_type = self.check_prediction_time()
        if not run_type:
            logger.warning("Not running predictions outside optimal times")
            return None
        
        try:
            # Get current time in Brussels
            brussels_tz = pytz.timezone('Europe/Brussels')
            now = pd.Timestamp.now(tz=brussels_tz)
            
            # Set target date based on run type
            if run_type == "evening":
                target_date = now.date() + timedelta(days=1)
                prediction_type = "Day-Ahead"
            else:  # morning run
                target_date = now.date()
                prediction_type = "Intraday Update"
            
            # Load model and scalers
            model, price_scaler, feature_scaler = self.load_model_and_scalers(country_code)
            
            # Load historical data for the sequence
            historical_data = pd.read_csv(
                self.processed_data_dir / f'processed_data_{country_code}.csv',
                index_col=0,
                parse_dates=True
            )
            
            # Fetch and prepare forecasts
            forecasts, _ = self.fetch_tomorrow_forecasts(country_code)
            features = self.prepare_features(forecasts)
            
            # Get the latest sequence of historical data
            latest_data = historical_data.iloc[-self.sequence_length:]
            
            # Scale historical features
            feature_columns = [col for col in historical_data.columns if col != 'price']
            latest_price = latest_data['price'].values.reshape(-1, 1)
            latest_features = feature_scaler.transform(latest_data[feature_columns])
            
            # Scale tomorrow's features
            tomorrow_features = feature_scaler.transform(features[feature_columns])
            
            # Create input sequence
            latest_data_scaled = pd.DataFrame(
                np.hstack([price_scaler.transform(latest_price), latest_features]),
                index=latest_data.index,
                columns=['price'] + feature_columns
            )
            
            # Create sequence
            X = self.create_sequences(latest_data_scaled, self.sequence_length)
            
            # Generate prediction
            y_pred_scaled = model.predict(X)
            
            # Inverse transform prediction
            y_pred = price_scaler.inverse_transform(y_pred_scaled)
            
            # Create predictions DataFrame with metadata
            predictions_df = pd.DataFrame({
                'timestamp': features.index,
                'predicted': y_pred.flatten(),
                'prediction_made': now,
                'prediction_type': prediction_type,
                'run_type': run_type,
                'forecast_basis': 'Evening Forecast' if run_type == "evening" else 'Morning Update'
            })
            predictions_df.set_index('timestamp', inplace=True)
            
            # Save predictions with date in filename
            output_file = self.dayahead_dir / f'prediction_{country_code}_{target_date.strftime("%Y%m%d")}_{run_type}.csv'
            predictions_df.to_csv(output_file)
            
            # Update latest prediction symlink
            latest_link = self.results_dir / f'dayahead_predictions_{country_code}.csv'
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(output_file)
            
            logger.info(f"Saved {prediction_type} predictions to {output_file}")
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error generating predictions for {country_code}: {str(e)}")
            raise

def main():
    predictor = DayAheadPredictor()
    
    # Generate predictions for all available countries
    model_files = list(Path('models').glob('lstm_model_*.keras'))
    countries = [f.stem.split('lstm_model_')[1] for f in model_files]
    
    for country_code in countries:
        try:
            predictions = predictor.predict_next_day(country_code)
            if predictions is not None:
                logger.info(f"Successfully generated day-ahead predictions for {country_code}")
                print(f"\nDay-ahead predictions for {country_code}:")
                print(predictions)
            
        except Exception as e:
            logger.error(f"Failed to generate predictions for {country_code}: {str(e)}")
            continue

if __name__ == "__main__":
    main()