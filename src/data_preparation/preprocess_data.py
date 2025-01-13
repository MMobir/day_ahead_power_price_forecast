import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, raw_data_dir='data/raw', processed_data_dir='data/processed'):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self, country_code='DE_LU'):
        """Load raw data files for a given country"""
        logger.info(f"Loading raw data for {country_code}")
        
        try:
            # Load prices (hourly data)
            prices = pd.read_csv(
                self.raw_data_dir / f'prices_{country_code}.csv',
                index_col=0,  # First column is index
                parse_dates=True,
                date_parser=lambda x: pd.to_datetime(x, utc=True)
            )
            prices.columns = ['price']  # Column '0' to 'price'
            
            # Load load forecast (15-min data)
            load_forecast = pd.read_csv(
                self.raw_data_dir / f'load_forecast_{country_code}.csv',
                index_col=0,
                parse_dates=True,
                date_parser=lambda x: pd.to_datetime(x, utc=True)
            )
            load_forecast.columns = ['load_forecast']  # Column 'Forecasted Load' to 'load_forecast'
            
            # Load wind/solar forecast
            wind_solar_forecast = pd.read_csv(
                self.raw_data_dir / f'wind_solar_forecast_{country_code}.csv',
                index_col=0,
                parse_dates=True,
                date_parser=lambda x: pd.to_datetime(x, utc=True)
            )
            
            # Convert all timestamps to UTC
            for df in [prices, load_forecast, wind_solar_forecast]:
                if df.index.tz is not None:
                    df.index = df.index.tz_convert('UTC')
                else:
                    df.index = df.index.tz_localize('UTC')
            
            # Resample to hourly frequency
            load_forecast = load_forecast.resample('h').mean()
            wind_solar_forecast = wind_solar_forecast.resample('h').mean()
            
            logger.info(f"Loaded {len(prices)} price records")
            logger.info(f"Loaded {len(load_forecast)} load forecast records")
            logger.info(f"Loaded {len(wind_solar_forecast)} wind/solar forecast records")
            
            return prices, load_forecast, wind_solar_forecast
            
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise
        
    def create_features(self, df):
        """Create time-based features"""
        df = df.copy()
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        # Create cyclical features for hour and month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        return df
        
    def create_lagged_features(self, df, price_col='price', n_lags=[24, 48, 168]):
        """Create lagged price features"""
        df = df.copy()
        
        for lag in n_lags:
            df[f'price_lag_{lag}h'] = df[price_col].shift(lag)
            
        # Add rolling statistics
        df['price_rolling_mean_24h'] = df[price_col].rolling(window=24).mean()
        df['price_rolling_std_24h'] = df[price_col].rolling(window=24).std()
        
        return df
    
    def align_and_combine_data(self, prices, load_forecast, wind_solar_forecast):
        """Align all data to the same timestamp index"""
        logger.info("Aligning and combining datasets")
        
        # Combine all features
        df = pd.DataFrame(index=prices.index)
        df['price'] = prices['price']
        df['load_forecast'] = load_forecast['load_forecast']
        
        # Handle different wind/solar column names
        if 'Wind Offshore' in wind_solar_forecast.columns:
            df['wind_offshore'] = wind_solar_forecast['Wind Offshore']
        else:
            df['wind_offshore'] = 0  # Set to 0 if no offshore wind
        
        if 'Wind Onshore' in wind_solar_forecast.columns:
            df['wind_onshore'] = wind_solar_forecast['Wind Onshore']
        elif 'Wind' in wind_solar_forecast.columns:  # Some countries might just have 'Wind'
            df['wind_onshore'] = wind_solar_forecast['Wind']
            df['wind_offshore'] = 0
        else:
            logger.warning("No wind generation columns found")
            df['wind_onshore'] = 0
        
        if 'Solar' in wind_solar_forecast.columns:
            df['solar'] = wind_solar_forecast['Solar']
        else:
            logger.warning("No solar generation column found")
            df['solar'] = 0
        
        # Calculate total wind generation
        df['total_wind_forecast'] = df['wind_offshore'] + df['wind_onshore']
        
        # Calculate renewable ratio
        df['renewable_ratio'] = (df['total_wind_forecast'] + df['solar']) / df['load_forecast']
        
        # Forward fill any missing values in forecasts
        df = df.ffill()  # Using ffill() instead of fillna(method='ffill') to address the warning
        
        return df
    
    def prepare_data(self, country_code='DE_LU'):
        """Main method to prepare the dataset"""
        # Load raw data
        prices, load_forecast, wind_solar_forecast = self.load_raw_data(country_code)
        
        # Combine all data
        df = self.align_and_combine_data(prices, load_forecast, wind_solar_forecast)
        
        # Create features
        df = self.create_features(df)
        df = self.create_lagged_features(df)
        
        # Remove rows with NaN values (due to lagged features)
        df = df.dropna()
        
        # Save processed data
        output_file = self.processed_data_dir / f'processed_data_{country_code}.csv'
        df.to_csv(output_file)
        logger.info(f"Processed data saved to {output_file}")
        
        return df
    
    def get_available_countries(self):
        """Get list of countries from price files in raw data directory"""
        price_files = list(self.raw_data_dir.glob('prices_*.csv'))
        countries = [f.stem.split('_', 1)[1] for f in price_files]
        return sorted(list(set(countries)))  # Remove duplicates and sort
    
    def prepare_all_countries(self):
        """Prepare data for all available countries"""
        countries = self.get_available_countries()
        logger.info(f"Found data for countries: {countries}")
        
        for country_code in countries:
            logger.info(f"\nProcessing data for {country_code}")
            try:
                df = self.prepare_data(country_code)
                logger.info(f"Successfully processed data for {country_code}")
            except Exception as e:
                logger.error(f"Error processing data for {country_code}: {str(e)}")
                continue

def main():
    preprocessor = DataPreprocessor()
    preprocessor.prepare_all_countries()

if __name__ == "__main__":
    main()