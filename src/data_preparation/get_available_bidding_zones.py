"""
Script to discover which ENTSO-E bidding zones have available day-ahead price data.
Tests all area codes and prints the ones that successfully return data.
"""

from entsoe import EntsoePandasClient, Area
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import pytz

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_area(client, area_code, start, end):
    """Test if an area has day-ahead price data"""
    try:
        # Get timezone for the area
        area = Area(area_code)
        tz = pytz.timezone(area.tz) if area.tz else pytz.UTC
        
        # Convert to pandas Timestamp with timezone
        start = pd.Timestamp(start).tz_convert(tz)
        end = pd.Timestamp(end).tz_convert(tz)
        
        prices = client.query_day_ahead_prices(area_code, start=start, end=end)
        if prices is not None and not prices.empty:
            logger.info(f"✅ {area_code} ({area.meaning}): Data available")
            logger.info(f"Sample data:\n{prices.head()}\n")
            return True
        else:
            logger.warning(f"❌ {area_code} ({area.meaning}): No data")
            return False
    except Exception as e:
        logger.error(f"❌ {area_code} ({area.meaning}): Error - {str(e)}")
        return False

def main():
    # Initialize API client
    client = EntsoePandasClient(api_key=os.getenv('ENTSOE_API_KEY'))
    
    # Set time range for yesterday (in UTC)
    end = pd.Timestamp.now(tz='UTC').normalize()  # normalize() sets time to midnight
    start = end - pd.Timedelta(days=1)
    
    logger.info(f"Testing all ENTSO-E areas for day-ahead prices from {start} to {end}")
    
    # Test all areas
    working_areas = []
    # Get all available areas from the ENTSO-E API
    for area in Area:
        if test_area(client, area.code, start, end):
            working_areas.append({
                'code': area.code,
                'short_code': area.name,
                'name': area.meaning,
                'timezone': area.tz
            })
    
    # Print summary
    logger.info("\n=== Summary of Available Bidding Zones ===")
    df = pd.DataFrame(working_areas)
    logger.info(f"\n{df.to_string()}")
    
    # Save results
    df.to_csv('data/available_bidding_zones.csv', index=False)
    logger.info("\nResults saved to available_bidding_zones.csv")

if __name__ == "__main__":
    main()