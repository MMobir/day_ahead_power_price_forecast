import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime, timedelta
import plotly.express as px
import pytz

# Set page config
st.set_page_config(
    page_title="Energy Price Forecasting Platform",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

def load_data(country_code):
    """Load predictions and metrics for a specific country"""
    project_root = Path(__file__).parents[2]
    results_dir = project_root / 'results'
    
    # Load predictions
    predictions_file = results_dir / f'predictions_{country_code}.csv'
    if not predictions_file.exists():
        raise FileNotFoundError(f"No predictions found for {country_code}. Please run predictions first.")
    
    predictions = pd.read_csv(predictions_file, index_col=0, parse_dates=True)
    
    # Load metrics
    metrics_file = results_dir / f'metrics_{country_code}.csv'
    if not metrics_file.exists():
        raise FileNotFoundError(f"No metrics found for {country_code}. Please run predictions first.")
    
    metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
    
    return predictions, metrics

def create_prediction_plot(data, dayahead_data=None, past_performance=None, country_code=None, start_date=None, end_date=None):
    """Create an interactive plot using plotly"""
    if start_date and end_date:
        mask = (data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))
        data = data[mask]
    
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['actual'],
        name='Actual',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['predicted'],
        name='Historical Prediction',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    # Add past 7-day performance if available
    if past_performance is not None:
        fig.add_trace(go.Scatter(
            x=past_performance.index,
            y=past_performance['predicted'],
            name='Past Day-Ahead Predictions',
            line=dict(color='#2ca02c', width=2)
        ))
    
    # Add day-ahead predictions if available
    if dayahead_data is not None:
        fig.add_trace(go.Scatter(
            x=dayahead_data.index,
            y=dayahead_data['predicted'],
            name='Tomorrow\'s Forecast',
            line=dict(color='#d62728', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f'Price Forecast - {country_code}',
        xaxis_title='Date',
        yaxis_title='Price (EUR/MWh)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_error_distribution(data):
    """Create error distribution plot"""
    errors = data['predicted'] - data['actual']
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=50,
        name='Error Distribution'
    ))
    
    fig.update_layout(
        title='Forecast Error Distribution',
        xaxis_title='Error (EUR/MWh)',
        yaxis_title='Count',
        template='plotly_white',
        showlegend=False,
        height=400
    )
    
    return fig

def create_scatter_plot(data):
    """Create scatter plot of predicted vs actual values"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['actual'],
        y=data['predicted'],
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.6,
            color='#1f77b4'
        ),
        name='Predictions'
    ))
    
    # Add perfect prediction line
    min_val = min(min(data['actual']), min(data['predicted']))
    max_val = max(max(data['actual']), max(data['predicted']))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title='Predicted vs Actual Prices',
        xaxis_title='Actual Price (EUR/MWh)',
        yaxis_title='Predicted Price (EUR/MWh)',
        template='plotly_white',
        height=400
    )
    
    return fig

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_country' not in st.session_state:
        st.session_state.current_country = None
    if 'date_range' not in st.session_state:
        st.session_state.date_range = None
    if 'date_mode' not in st.session_state:
        st.session_state.date_mode = 'single'

def filter_data_by_date(data, date_selection, mode='single'):
    """Filter data based on date selection and mode"""
    if mode == 'single':
        return data[data.index.date == date_selection]
    else:
        start_date, end_date = date_selection
        return data[
            (data.index.date >= start_date) & 
            (data.index.date <= end_date)
        ]

def format_predictions_table(predictions_df):
    """Format predictions table for better visualization"""
    df = predictions_df.copy()
    df['Hour'] = df.index.strftime('%H:00')
    df['Price (€/MWh)'] = df['predicted'].round(2)
    df['Peak/Off-Peak'] = df.index.hour.map(lambda x: 'Peak' if 8 <= x <= 20 else 'Off-Peak')
    
    # Add conditional formatting
    return df[['Hour', 'Price (€/MWh)', 'Peak/Off-Peak']].style\
        .background_gradient(subset=['Price (€/MWh)'])\
        .format({'Price (€/MWh)': '{:.2f}'})

def get_market_status():
    """Get current market status and time to next closure"""
    brussels_tz = pytz.timezone('Europe/Brussels')
    now = pd.Timestamp.now(tz=brussels_tz)
    market_close = pd.Timestamp.combine(
        now.date() + timedelta(days=1),
        pd.Timestamp('12:00').time()
    ).tz_localize(brussels_tz)
    
    if now.hour >= 12:
        market_close = market_close + timedelta(days=1)
    
    time_to_close = market_close - now
    hours = int(time_to_close.total_seconds() // 3600)
    minutes = int((time_to_close.total_seconds() % 3600) // 60)
    
    return f"{hours:02d}:{minutes:02d}"

def group_zones_by_region(active_zones):
    """Group bidding zones by region"""
    regions = {
        'Central Western Europe': ['DE_LU', 'FR', 'NL', 'BE', 'AT', 'CH'],
        'Nordics': ['DK_1', 'DK_2', 'NO_1', 'NO_2', 'NO_3', 'NO_4', 'NO_5', 'SE_1', 'SE_4', 'FI'],
        'Baltics': ['EE', 'LV', 'LT'],
        'Central Eastern Europe': ['CZ', 'HU', 'PL', 'RO', 'SI', 'HR'],
        'Southern Europe': ['IT_CALA', 'IT_CNOR', 'IT_CSUD', 'IT_NORD', 'IT_SARD', 'IT_SICI', 'IT_SUD', 'GR', 'ES', 'PT'],
        'South Eastern Europe': ['BG', 'RS', 'ME']
    }
    
    # Create a mapping of zones to their display names
    zone_display = {
        row['short_code']: f"{row['name']} ({row['short_code']})"
        for _, row in active_zones.iterrows()
    }
    
    # Group active zones by region
    grouped_zones = {}
    for region, codes in regions.items():
        # Filter for active zones in this region
        region_zones = {
            zone_display[code]: code
            for code in codes
            if code in zone_display
        }
        if region_zones:  # Only add region if it has active zones
            grouped_zones[region] = region_zones
    
    return grouped_zones

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("⚡ Energy Price Forecasting Platform")
    
    # Get project root and read available zones
    project_root = Path(__file__).parents[2]
    bidding_zones_df = pd.read_csv(project_root / 'data' / 'available_bidding_zones.csv')
    active_zones = bidding_zones_df[bidding_zones_df['active'] == 1]
    
    # Sidebar
    st.sidebar.header("Market Selection")
    
    # Group zones by region
    grouped_zones = group_zones_by_region(active_zones)
    
    # Region selection
    selected_region = st.sidebar.selectbox(
        "Select Region",
        options=list(grouped_zones.keys())
    )
    
    # Zone selection within region
    selected_zone_name = st.sidebar.selectbox(
        "Select Market Zone",
        options=list(grouped_zones[selected_region].keys()) if selected_region else []
    )
    
    if selected_zone_name:
        country_code = grouped_zones[selected_region][selected_zone_name]
        
        try:
            # Load data
            predictions, metrics = load_data(country_code)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MAE", f"{metrics['MAE']:.2f} €/MWh")
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.2f} €/MWh")
            with col3:
                st.metric("R²", f"{metrics['R2']:.3f}")
            with col4:
                mape = metrics['MAPE']
                if np.isfinite(mape):
                    st.metric("MAPE", f"{mape:.2f}%")
                else:
                    st.metric("MAPE", "N/A")
            
            # Date selection
            st.sidebar.header("Date Range")
            date_mode = st.sidebar.radio("Select Date Mode", ['Last 7 Days', 'Last 30 Days', 'Custom Range'])
            
            if date_mode == 'Custom Range':
                start_date = st.sidebar.date_input("Start Date", predictions.index.min())
                end_date = st.sidebar.date_input("End Date", predictions.index.max())
                if start_date and end_date:
                    mask = (predictions.index.date >= start_date) & (predictions.index.date <= end_date)
                    filtered_predictions = predictions[mask]
            elif date_mode == 'Last 7 Days':
                end_date = predictions.index.max()
                start_date = end_date - pd.Timedelta(days=7)
                mask = (predictions.index >= start_date)
                filtered_predictions = predictions[mask]
            else:  # Last 30 Days
                end_date = predictions.index.max()
                start_date = end_date - pd.Timedelta(days=30)
                mask = (predictions.index >= start_date)
                filtered_predictions = predictions[mask]
            
            # Main plots
            st.plotly_chart(create_prediction_plot(
                filtered_predictions, 
                country_code=country_code
            ), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_error_distribution(filtered_predictions), use_container_width=True)
            with col2:
                st.plotly_chart(create_scatter_plot(filtered_predictions), use_container_width=True)
            
            # Data table
            st.subheader("Detailed Predictions")
            st.dataframe(
                format_predictions_table(filtered_predictions),
                use_container_width=True
            )
            
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    else:
        st.warning("Please select a market zone to view predictions.")

if __name__ == "__main__":
    main() 