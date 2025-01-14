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
    results_dir = Path('results')
    dayahead_dir = results_dir / 'dayahead'
    
    # Load historical predictions
    predictions_file = results_dir / f'predictions_{country_code}.csv'
    if not predictions_file.exists():
        raise FileNotFoundError(f"No predictions found for {country_code}. Please run predictions first.")
    
    predictions = pd.read_csv(predictions_file, index_col=0, parse_dates=True)
    
    # Load latest day-ahead predictions
    dayahead_file = results_dir / f'dayahead_predictions_{country_code}.csv'
    dayahead_predictions = None
    if dayahead_file.exists():
        dayahead_predictions = pd.read_csv(dayahead_file, index_col=0, parse_dates=True)
    
    # Load past 7 days of day-ahead predictions for performance analysis
    past_predictions = []
    today = pd.Timestamp.now(tz='Europe/Brussels').date()
    for i in range(1, 8):  # Last 7 days
        date = today - timedelta(days=i)
        file = dayahead_dir / f'prediction_{country_code}_{date.strftime("%Y%m%d")}.csv'
        if file.exists():
            pred = pd.read_csv(file, index_col=0, parse_dates=True)
            past_predictions.append(pred)
    
    past_performance = pd.concat(past_predictions) if past_predictions else None
    
    # Load metrics
    metrics_file = results_dir / f'metrics_{country_code}.csv'
    if not metrics_file.exists():
        raise FileNotFoundError(f"No metrics found for {country_code}. Please run predictions first.")
    
    metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
    
    return predictions, dayahead_predictions, past_performance, metrics

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

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("⚡ Energy Price Forecasting Platform")
    
    # Top bar with market information
    market_col1, market_col2, market_col3 = st.columns(3)
    with market_col1:
        st.metric("Next Market Closure", "12:00 CET Tomorrow", delta=get_market_status())
    with market_col2:
        st.metric("Current Time (CET)", pd.Timestamp.now(tz='Europe/Brussels').strftime('%H:%M:%S'))
    with market_col3:
        st.metric("Market Status", "Open" if pd.Timestamp.now(tz='Europe/Brussels').hour < 12 else "Closed for Tomorrow")
    
    # Sidebar
    st.sidebar.header("Market Selection")
    
    # Get available countries
    available_countries = [
        path.stem.split('predictions_')[1] 
        for path in Path('results').glob('predictions_*.csv')
    ]
    
    if not available_countries:
        st.error("No predictions available. Please run predictions first.")
        return
    
    # Country selection
    country_code = st.sidebar.selectbox(
        "Select Market",
        available_countries,
        format_func=lambda x: {
            'DE_LU': '🇩🇪 Germany-Luxembourg',
            'FR': '🇫🇷 France',
            'ES': '🇪🇸 Spain',
            'IT_NORTH': '🇮🇹 Italy (North)',
            'NL': '🇳🇱 Netherlands'
        }.get(x, x)
    )
    
    try:
        # Load data
        predictions, dayahead_predictions, past_performance, metrics = load_data(country_code)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Market Overview", "Performance Analysis", "Historical Data"])
        
        with tab1:
            # Show prediction status and metadata
            if dayahead_predictions is not None:
                status_col1, status_col2 = st.columns(2)
                with status_col1:
                    st.success("✅ Latest predictions available")
                    st.markdown(f"**Forecast Period:** {dayahead_predictions.index[0].strftime('%Y-%m-%d')}")
                    st.markdown(f"**Generated:** {dayahead_predictions['prediction_made'].iloc[0].strftime('%Y-%m-%d %H:%M')} CET")
                    st.markdown(f"**Type:** {dayahead_predictions['prediction_type'].iloc[0]}")
                
                # Show key price statistics
                st.markdown("### Price Forecast Summary")
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                with summary_col1:
                    st.metric(
                        "Peak Price (08-20)",
                        f"{dayahead_predictions['predicted'][dayahead_predictions.index.hour.isin(range(8,21))].max():.2f} €/MWh"
                    )
                with summary_col2:
                    st.metric(
                        "Base Price (00-24)",
                        f"{dayahead_predictions['predicted'].mean():.2f} €/MWh"
                    )
                with summary_col3:
                    st.metric(
                        "Off-Peak Price",
                        f"{dayahead_predictions['predicted'][~dayahead_predictions.index.hour.isin(range(8,21))].mean():.2f} €/MWh"
                    )
                with summary_col4:
                    st.metric(
                        "Price Volatility",
                        f"{dayahead_predictions['predicted'].std():.2f} €/MWh"
                    )
                
                # Show hourly predictions
                st.markdown("### Hourly Price Forecast")
                st.dataframe(format_predictions_table(dayahead_predictions), use_container_width=True)
                
                # Show prediction plot
                st.plotly_chart(
                    create_prediction_plot(predictions, dayahead_predictions, past_performance, country_code),
                    use_container_width=True
                )
            else:
                st.warning("⚠️ No predictions available for tomorrow yet. Next update at 18:15 CET.")
        
        with tab2:
            st.header("Forecast Performance Analysis")
            
            # Show overall model metrics
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                st.metric("MAE", f"{metrics['MAE']:.2f} €/MWh")
            with metrics_col2:
                st.metric("RMSE", f"{metrics['RMSE']:.2f} €/MWh")
            with metrics_col3:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            with metrics_col4:
                st.metric("R²", f"{metrics['R2']:.3f}")
            
            # Show recent performance if available
            if past_performance is not None:
                st.markdown("### Recent Performance")
                recent_mae = mean_absolute_error(
                    predictions[predictions.index.isin(past_performance.index)]['actual'],
                    past_performance['predicted']
                )
                recent_mape = np.mean(np.abs(
                    (predictions[predictions.index.isin(past_performance.index)]['actual'] - 
                     past_performance['predicted']) / 
                    predictions[predictions.index.isin(past_performance.index)]['actual']
                )) * 100
                
                perf_col1, perf_col2 = st.columns(2)
                with perf_col1:
                    st.metric(
                        "7-Day Accuracy",
                        f"{(100 - recent_mape):.1f}%",
                        delta=f"{(100 - recent_mape) - (100 - float(metrics['MAPE'])):.1f}"
                    )
                with perf_col2:
                    st.metric(
                        "7-Day MAE",
                        f"{recent_mae:.2f} €/MWh",
                        delta=f"{metrics['MAE'] - recent_mae:.2f}"
                    )
            
            # Error analysis
            st.markdown("### Error Analysis")
            error_col1, error_col2 = st.columns(2)
            with error_col1:
                st.plotly_chart(create_error_distribution(predictions), use_container_width=True)
            with error_col2:
                st.plotly_chart(create_scatter_plot(predictions), use_container_width=True)
        
        with tab3:
            st.header("Historical Data Analysis")
            
            # Date range selector
            date_range = st.date_input(
                "Select Date Range",
                value=(predictions.index[-30].date(), predictions.index[-1].date()),
                min_value=predictions.index[0].date(),
                max_value=predictions.index[-1].date()
            )
            
            if isinstance(date_range, tuple):
                start_date, end_date = date_range
                mask = (predictions.index.date >= start_date) & (predictions.index.date <= end_date)
                historical_data = predictions[mask]
                
                st.plotly_chart(
                    create_prediction_plot(historical_data, None, None, country_code),
                    use_container_width=True
                )
                
                # Show statistics for selected period
                hist_col1, hist_col2, hist_col3, hist_col4 = st.columns(4)
                with hist_col1:
                    st.metric("Average Price", f"{historical_data['actual'].mean():.2f} €/MWh")
                with hist_col2:
                    st.metric("Max Price", f"{historical_data['actual'].max():.2f} €/MWh")
                with hist_col3:
                    st.metric("Min Price", f"{historical_data['actual'].min():.2f} €/MWh")
                with hist_col4:
                    st.metric("Volatility", f"{historical_data['actual'].std():.2f} €/MWh")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 