"""
This script fetches data from the ENTSO-E Transparency Platform API (https://transparency.entsoe.eu/).
The following data is fetched for each bidding zone:
1. Day-ahead electricity prices
2. Load forecasts
3. Wind and solar generation forecasts

Required environment variables:
- ENTSOE_API_KEY: Your API key from the ENTSO-E Transparency Platform
"""

from entsoe import EntsoePandasClient
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
from pathlib import Path
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def retry_api_call(func, max_retries=3, retry_delay=5):
    """Retry API calls with exponential backoff, max 3 attempts"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed after {max_retries} attempts, moving on...")
                return None
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time} seconds... Error: {str(e)}")
            time.sleep(wait_time)

def fetch_historical_prices(client, country_code, start_date, end_date):
    """Fetch day-ahead electricity prices from ENTSO-E"""
    try:
        def _fetch():
            return client.query_day_ahead_prices(country_code, start=start_date, end=end_date)
        return retry_api_call(_fetch)
    except Exception as e:
        logger.error(f"Error fetching day-ahead prices: {str(e)}")
        return None

def fetch_load_forecast(client, country_code, start_date, end_date):
    """Fetch day-ahead load forecasts from ENTSO-E"""
    try:
        def _fetch():
            return client.query_load_forecast(country_code, start=start_date, end=end_date, process_type="A01")
        return retry_api_call(_fetch)
    except Exception as e:
        logger.error(f"Error fetching load forecast: {str(e)}")
        return None

def fetch_wind_solar_forecast(client, country_code, start_date, end_date):
    """Fetch wind and solar generation forecasts from ENTSO-E"""
    try:
        def _fetch():
            return client.query_wind_and_solar_forecast(country_code, start=start_date, end=end_date)
        return retry_api_call(_fetch)
    except Exception as e:
        logger.error(f"Error fetching wind/solar forecast: {str(e)}")
        return None

def fetch_data_for_country(client, country_code, start_date, end_date):
    """Fetch all required data for a country"""
    logger.info(f"Fetching data for {country_code} from {start_date} to {end_date}")
    
    # Fetch all data types
    prices = fetch_historical_prices(client, country_code, start_date, end_date)
    load = fetch_load_forecast(client, country_code, start_date, end_date)
    forecast = fetch_wind_solar_forecast(client, country_code, start_date, end_date)
    
    return {
        'prices': prices,
        'load': load,
        'forecast': forecast
    }

def process_and_save_data(raw_data, country_code, project_root):
    """Process and save data for a country"""
    try:
        # Extract data
        prices = raw_data['prices']
        load = raw_data['load']
        forecast = raw_data['forecast']
        
        # Create base dataframe with prices
        df = pd.DataFrame(index=prices.index)
        df['price'] = prices
        
        # Add load forecast
        df['load_forecast'] = load.reindex(df.index)
        
        # Add wind and solar forecasts
        if 'Wind Offshore' in forecast.columns:
            df['wind_offshore'] = forecast['Wind Offshore'].reindex(df.index)
        else:
            df['wind_offshore'] = 0
            
        if 'Wind Onshore' in forecast.columns:
            df['wind_onshore'] = forecast['Wind Onshore'].reindex(df.index)
        else:
            df['wind_onshore'] = 0
            
        if 'Solar' in forecast.columns:
            df['solar'] = forecast['Solar'].reindex(df.index)
        else:
            df['solar'] = 0
        
        # Calculate total wind forecast
        df['total_wind_forecast'] = df['wind_offshore'] + df['wind_onshore']
        
        # Calculate renewable ratio
        total_generation = df['total_wind_forecast'] + df['solar']
        df['renewable_ratio'] = total_generation / df['load_forecast']
        
        # Add time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        # Add cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Add price lags
        df['price_lag_24h'] = df['price'].shift(24)
        df['price_lag_48h'] = df['price'].shift(48)
        df['price_lag_168h'] = df['price'].shift(168)
        
        # Add rolling statistics
        df['price_rolling_mean_24h'] = df['price'].rolling(window=24).mean()
        df['price_rolling_std_24h'] = df['price'].rolling(window=24).std()
        
        # Save to CSV in project's data directory
        output_file = project_root / 'data' / f'{country_code}.csv'
        df.to_csv(output_file)
        logger.info(f"Saved data to {output_file}")
        
        # If data is older than 30 days, move it to historical_data
        latest_date = df.index.max()
        if (pd.Timestamp.now(tz='UTC') - latest_date).days > 30:
            historical_file = project_root / 'data' / 'historical_data' / f'{country_code}.csv'
            df.to_csv(historical_file)
            logger.info(f"Moved data to historical archive: {historical_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return False

def main():
    # Initialize API client
    client = EntsoePandasClient(api_key=os.getenv('ENTSOE_API_KEY'))
    
    # Get project root directory (2 levels up from this script)
    project_root = Path(__file__).parents[2]
    
    # Create historical_data directory if it doesn't exist
    (project_root / 'data' / 'historical_data').mkdir(parents=True, exist_ok=True)
    
    # Read available bidding zones from CSV
    bidding_zones_df = pd.read_csv(project_root / 'data' / 'available_bidding_zones.csv')
    # Filter for active zones only
    active_zones = bidding_zones_df[bidding_zones_df['active'] == 1]
    logger.info(f"Loaded {len(active_zones)} active bidding zones from CSV")
    
    # Get current time in UTC and set to previous 703 days
    now = pd.Timestamp.now(tz='UTC')
    end_date = now.normalize()  # Set to midnight
    start_date = end_date - timedelta(days=703)
    
    # Get list of already processed countries
    existing_files = set(f.stem for f in (project_root / 'data').glob('*.csv') 
                        if f.stem != 'available_bidding_zones')
    logger.info(f"Found existing data for: {', '.join(sorted(existing_files))}")
    
    for _, row in active_zones.iterrows():
        country_code = row['code']
        short_code = row['short_code']
            
        try:
            logger.info(f"Processing {short_code} ({country_code})")
            
            # Fetch data
            raw_data = fetch_data_for_country(client, country_code, start_date, end_date)
            
            # Process and save data
            if all(v is not None for v in raw_data.values()):
                success = process_and_save_data(raw_data, short_code, project_root)
                if success:
                    logger.info(f"Successfully processed data for {short_code}")
            else:
                logger.error(f"Missing data for {short_code}")
                
        except Exception as e:
            logger.error(f"Error processing {short_code}: {str(e)}")
            continue
            
        # Add a small delay between countries to avoid rate limiting
        time.sleep(2)

if __name__ == "__main__":
    main()