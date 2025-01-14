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

def retry_api_call(func, max_retries=5, retry_delay=10):
    """Retry API calls with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"API call failed, retrying in {wait_time} seconds... Error: {str(e)}")
            time.sleep(wait_time)

def fetch_historical_prices(client, country_code, start_date, end_date):
    """Fetch day-ahead electricity prices"""
    try:
        def _fetch():
            return client.query_day_ahead_prices(
                country_code,
                start=start_date,
                end=end_date
            )
        return retry_api_call(_fetch)
    except Exception as e:
        logger.error(f"Error fetching day-ahead prices: {str(e)}")
        return None

def fetch_actual_generation(client, country_code, start_date, end_date):
    """Fetch actual power generation data"""
    try:
        def _fetch():
            return client.query_generation(
                country_code,
                start=start_date,
                end=end_date,
                psr_type=None
            )
        return retry_api_call(_fetch)
    except Exception as e:
        logger.error(f"Error fetching generation data: {str(e)}")
        return None

def fetch_wind_solar_forecast(client, country_code, start_date, end_date):
    """Fetch wind and solar generation forecasts"""
    try:
        def _fetch():
            return client.query_wind_and_solar_forecast(
                country_code,
                start=start_date,
                end=end_date
            )
        return retry_api_call(_fetch)
    except Exception as e:
        logger.error(f"Error fetching wind/solar forecast: {str(e)}")
        return None

def fetch_load_forecast(client, country_code, start_date, end_date):
    """Fetch day-ahead load forecast"""
    try:
        def _fetch():
            return client.query_load_forecast(
                country_code,
                start=start_date,
                end=end_date,
                process_type="A01"
            )
        return retry_api_call(_fetch)
    except Exception as e:
        logger.error(f"Error fetching load forecast: {str(e)}")
        return None

def process_and_save_data(raw_data, processed_data_dir, country_code):
    """Process raw data and save to processed directory"""
    try:
        # Extract data
        prices = raw_data['prices']
        generation = raw_data['generation']
        forecast = raw_data['forecast']
        load = raw_data['load']
        
        # Process generation data
        if isinstance(generation, pd.DataFrame):
            wind = generation.get('Wind Onshore', 0) + generation.get('Wind Offshore', 0)
            solar = generation.get('Solar', 0)
        else:
            wind = pd.Series(0, index=prices.index)
            solar = pd.Series(0, index=prices.index)
        
        # Process forecast data
        wind_forecast = forecast.get('Wind Onshore', 0) + forecast.get('Wind Offshore', 0)
        solar_forecast = forecast.get('Solar', 0)
        
        # Combine all data
        data = pd.DataFrame({
            'price': prices,
            'load': load,
            'wind': wind,
            'solar': solar,
            'wind_forecast': wind_forecast,
            'solar_forecast': solar_forecast
        })
        
        # Add derived features
        data['total_wind'] = data['wind']
        data['renewable_ratio'] = (data['wind'] + data['solar']) / data['load']
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        data['weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)
        
        # Add cyclical features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
        data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
        data['month_cos'] = np.cos(2 * np.pi * data['month']/12)
        
        # Save processed data
        processed_file = processed_data_dir / f'processed_data_{country_code}.csv'
        if processed_file.exists():
            # If file exists, append new data
            existing_data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
            combined_data = pd.concat([existing_data, data])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data.sort_index(inplace=True)
            combined_data.to_csv(processed_file)
            logger.info(f"Updated processed data for {country_code}")
            logger.info(f"Data range: {combined_data.index[0]} to {combined_data.index[-1]}")
        else:
            # If file doesn't exist, create new
            data.to_csv(processed_file)
            logger.info(f"Created new processed data file for {country_code}")
        
        return True
    except Exception as e:
        logger.error(f"Error processing data for {country_code}: {str(e)}")
        return False

def fetch_data_for_country(client, country_code, start_date, end_date, chunk_size=timedelta(days=30)):
    """Fetch data for a country in chunks to handle API limits"""
    logger.info(f"Fetching data for {country_code} from {start_date} to {end_date}")
    
    current_start = start_date
    raw_data = {
        'prices': [],
        'generation': [],
        'forecast': [],
        'load': []
    }
    
    while current_start < end_date:
        current_end = min(current_start + chunk_size, end_date)
        logger.info(f"Fetching chunk: {current_start} to {current_end}")
        
        # Fetch all data types
        prices = fetch_historical_prices(client, country_code, current_start, current_end)
        generation = fetch_actual_generation(client, country_code, current_start, current_end)
        forecast = fetch_wind_solar_forecast(client, country_code, current_start, current_end)
        load = fetch_load_forecast(client, country_code, current_start, current_end)
        
        # Append non-None results
        if prices is not None:
            raw_data['prices'].append(prices)
        if generation is not None:
            raw_data['generation'].append(generation)
        if forecast is not None:
            raw_data['forecast'].append(forecast)
        if load is not None:
            raw_data['load'].append(load)
        
        current_start = current_end
    
    # Combine chunks
    return {
        'prices': pd.concat(raw_data['prices']) if raw_data['prices'] else None,
        'generation': pd.concat(raw_data['generation']) if raw_data['generation'] else None,
        'forecast': pd.concat(raw_data['forecast']) if raw_data['forecast'] else None,
        'load': pd.concat(raw_data['load']) if raw_data['load'] else None
    }

def main():
    # Initialize API client
    client = EntsoePandasClient(api_key=os.getenv('ENTSOE_API_KEY'))
    
    # Set up directories
    raw_data_dir = Path('data/raw')
    processed_data_dir = Path('data/processed')
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define countries
    country_codes = [
        'DE_LU',  # Germany-Luxembourg
        'FR',     # France
        'ES',     # Spain
        'NL'      # Netherlands
    ]
    
    # Get current time in Brussels timezone
    brussels_tz = pytz.timezone('Europe/Brussels')
    now = pd.Timestamp.now(tz=brussels_tz)
    
    for country_code in country_codes:
        try:
            # Check if processed file exists
            processed_file = processed_data_dir / f'processed_data_{country_code}.csv'
            if processed_file.exists():
                # If file exists, only fetch missing data
                existing_data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
                start_date = existing_data.index[-1] - timedelta(hours=1)  # Small overlap to ensure no gaps
                end_date = now
                logger.info(f"Updating existing data for {country_code} from {start_date}")
            else:
                # If file doesn't exist, fetch 2 years of historical data
                end_date = now
                start_date = end_date - timedelta(days=730)
                logger.info(f"Fetching historical data for {country_code}")
            
            # Fetch data
            raw_data = fetch_data_for_country(client, country_code, start_date, end_date)
            
            # Process and save data
            if all(v is not None for v in raw_data.values()):
                success = process_and_save_data(raw_data, processed_data_dir, country_code)
                if success:
                    logger.info(f"Successfully processed data for {country_code}")
                else:
                    logger.error(f"Failed to process data for {country_code}")
            else:
                logger.error(f"Missing data for {country_code}")
                
        except Exception as e:
            logger.error(f"Error processing {country_code}: {str(e)}")
            continue

if __name__ == "__main__":
    main()