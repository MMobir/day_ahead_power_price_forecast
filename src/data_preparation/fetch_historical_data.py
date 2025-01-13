from entsoe import EntsoePandasClient
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Load environment variables
load_dotenv()

def fetch_historical_prices(client, country_code, start_date, end_date):
    """Fetch day-ahead electricity prices"""
    try:
        prices = client.query_day_ahead_prices(
            country_code,
            start=start_date,
            end=end_date
        )
        return prices
    except Exception as e:
        print(f"Error fetching day-ahead prices: {str(e)}")
        return None

def fetch_actual_generation(client, country_code, start_date, end_date):
    """Fetch actual power generation data"""
    try:
        generation = client.query_generation(
            country_code,
            start=start_date,
            end=end_date,
            psr_type=None  # None to get all generation types
        )
        return generation
    except Exception as e:
        print(f"Error fetching generation data: {str(e)}")
        return None

def fetch_wind_solar_forecast(client, country_code, start_date, end_date):
    """Fetch wind and solar generation forecasts"""
    try:
        forecast = client.query_wind_and_solar_forecast(
            country_code,
            start=start_date,
            end=end_date
        )
        return forecast
    except Exception as e:
        print(f"Error fetching wind/solar forecast: {str(e)}")
        return None

def fetch_load_forecast(client, country_code, start_date, end_date):
    """Fetch day-ahead load forecast"""
    try:
        load = client.query_load_forecast(
            country_code,
            start=start_date,
            end=end_date,
            process_type="A01"  # Day-ahead
        )
        return load
    except Exception as e:
        print(f"Error fetching load forecast: {str(e)}")
        return None

def main():
    # Initialize API client
    client = EntsoePandasClient(api_key=os.getenv('ENTSOE_API_KEY'))
    
    # Define time range (2 years of historical data)
    end_date = pd.Timestamp.now(tz='Europe/Brussels')
    start_date = end_date - pd.Timedelta(days=730)
    
    # Define countries
    country_codes = [
        'DE_LU',  # Germany-Luxembourg
        'FR',     # France
        'ES',     # Spain
        'IT_NORTH',# Italy North
        'NL'      # Netherlands
    ]
    
    # Create raw data directory
    raw_data_dir = 'data/raw'
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Fetch data for each country
    for country_code in country_codes:
        print(f"\nFetching data for {country_code}...")
        
        # Fetch and save prices
        prices = fetch_historical_prices(client, country_code, start_date, end_date)
        if prices is not None:
            prices.to_csv(os.path.join(raw_data_dir, f'prices_{country_code}.csv'))
            
        # Fetch and save generation
        generation = fetch_actual_generation(client, country_code, start_date, end_date)
        if generation is not None:
            generation.to_csv(os.path.join(raw_data_dir, f'generation_{country_code}.csv'))
            
        # Fetch and save wind/solar forecast
        forecast = fetch_wind_solar_forecast(client, country_code, start_date, end_date)
        if forecast is not None:
            forecast.to_csv(os.path.join(raw_data_dir, f'wind_solar_forecast_{country_code}.csv'))
            
        # Fetch and save load forecast
        load = fetch_load_forecast(client, country_code, start_date, end_date)
        if load is not None:
            load.to_csv(os.path.join(raw_data_dir, f'load_forecast_{country_code}.csv'))

if __name__ == "__main__":
    main()