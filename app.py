import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load and process the weather data
@st.cache
def load_data():
    # Replace this with the actual path to your CSV file or load the dataset accordingly
    df = pd.read_csv('weather.csv')

    # Convert 'time' to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Convert temperature from Kelvin to Fahrenheit
    df['Ftemp'] = (df['Ktemp'] - 273.15) * (9/5) + 32
    
    # Extract year and month
    df['Year'] = df['time'].dt.year
    df['Month'] = df['time'].dt.month
    
    return df

df = load_data()

# Part A: Average Monthly Temperature Plot
st.title("Weather Data Visualization")
st.header("Part A: Average Temperature by Month")

# Allow the user to select the year to view
selected_year = st.slider('Select Year', min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=int(df['Year'].max()))

# Filter data for the selected year
df_year = df[df['Year'] == selected_year]

# Group by month and calculate the average temperature for the selected year
monthly_avg = df_year.groupby('Month')['Ftemp'].mean().reset_index()

# Create a Plotly line plot
fig = px.line(monthly_avg, x='Month', y='Ftemp', 
              labels={'Ftemp': 'Average Temperature (°F)', 'Month': 'Month'},
              title=f"Average Monthly Temperature in {selected_year} (°F)")

# Show the plot in the Streamlit app
st.plotly_chart(fig)

st.write("""
    This is a monthly average temperature display from 1950 to 2021. By sliding the bar above, you
    could choose the year you want to view. We do see that the yearly patterns are very similar, with 
    temperature peak around July and August.
""")

# Part B: First Year with Average Temperature Above 55°F
st.header("Part B: First Year with Average Temperature Above 55°F")

# Group by year and calculate the average temperature for each year
yearly_avg = df.groupby('Year')['Ftemp'].mean().reset_index()

# Find the first year where the average temperature exceeds 55°F
first_year_above_55 = yearly_avg[yearly_avg['Ftemp'] > 55]['Year'].min()

# Display the result
st.write(f"The first year when the average temperature exceeds 55°F is {first_year_above_55}.")

# Create the line chart of yearly average temperature
fig = px.line(yearly_avg, x='Year', y='Ftemp', title='Yearly Average Temperature (°F)', 
              labels={'Ftemp': 'Average Temperature (°F)', 'Year': 'Year'})

# Add a horizontal red line at 55°F
fig.add_hline(y=55, line_dash="dash", line_color="red", annotation_text="55°F", annotation_position="top right")

# Show the figure in the Streamlit app
st.plotly_chart(fig)

# Part C: Creative Visualization (Temperature Trends over Time)
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Assuming 'df' is your DataFrame already loaded with the weather data

# Time-Series Forecasting with SARIMA
st.header("Part C-1: Time-Series Prediction of Average Temperature with SARIMA")

# Convert the 'time' column to datetime if it's not already
df['time'] = pd.to_datetime(df['time'])

# Extract year and month and create a datetime index
df['YearMonth'] = df['time'].dt.to_period('M')

# Calculate the monthly average temperature
monthly_avg = df.groupby('YearMonth')['Ftemp'].mean()

# SARIMA Model
st.subheader("Fitting SARIMA Model")

# Define SARIMA model (p, d, q) and (P, D, Q, S)
# We use (p=1, d=1, q=1) for non-seasonal and (P=1, D=1, Q=1, S=12) for seasonal (12 months in a year)
sarima_model = SARIMAX(monthly_avg, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
sarima_model_fit = sarima_model.fit(disp=False)

# Forecast the next 36 months (extend the forecast for comparison)
forecast_steps = 36
forecast = sarima_model_fit.forecast(steps=forecast_steps)

# Create the forecasted date range
forecast_index = pd.date_range(start=monthly_avg.index[-1].end_time + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

# Actual temperatures for 2022 to 2024 (from Jan 2022 to Dec 2024)
actual_all = np.array([
    31.17, 33.66, 44.08, 50.67, 61.92, 70.79, 76.39, 74.59, 68.02, 55.18, 40.96, 33.12,  # 2022
    35.08, 36.34, 40.53, 51.28, 62.37, 68.99, 75.65, 74.35, 67.75, 56.03, 44.22, 39.76,  # 2023
    31.89, 40.82, 44.98, 53.64, 62.28, 71.80, 75.69, 73.99, 68.61, 58.96, 45.19, 38.25   # 2024
])

# Create a DataFrame with forecast and actual data for comparison
forecast_df = pd.DataFrame({
    'Date': forecast_index,
    'Forecasted Temp (°F)': forecast,
    'Actual Temp (°F)': actual_all
})

# Monthly data chart
st.subheader("Historical vs Forecasted Monthly Temperatures")

fig_monthly = go.Figure()

# Historical data (monthly averages)
fig_monthly.add_trace(go.Scatter(x=monthly_avg.index.astype(str), y=monthly_avg.values, mode='lines', name='Historical Data'))

# Forecasted data (next 36 months)
fig_monthly.add_trace(go.Scatter(x=forecast_index.astype(str), y=forecast, mode='lines+markers', name='Forecasted Data', line=dict(dash='dash')))

fig_monthly.update_layout(
    title='Historical and Forecasted Monthly Temperatures',
    xaxis_title='Date',
    yaxis_title='Average Temperature (°F)',
    showlegend=True,
    width=2600,
    height=600
)

st.plotly_chart(fig_monthly)

# Create the close-up chart comparing actual vs forecasted for 2022-2024
fig_comparison = go.Figure()

# Add actual temperatures as a red line with dots
fig_comparison.add_trace(go.Scatter(
    x=forecast_index, y=actual_all,
    mode='lines+markers', name='Actual Temperature', 
    line=dict(color='red'), marker=dict(symbol='circle', size=8)
))

# Add forecasted temperatures as a blue dashed line with dots
fig_comparison.add_trace(go.Scatter(
    x=forecast_index, y=forecast,
    mode='lines+markers', name='Forecasted Temperature', 
    line=dict(color='blue', dash='dash'), marker=dict(symbol='circle', size=8)
))

# Update layout for the close-up comparison chart
fig_comparison.update_layout(
    title="Close-up: Predicted vs Actual Monthly Temperatures for 2022-2024",
    xaxis_title="Month",
    yaxis_title="Temperature (°F)",
    width=800,
    height=400,
    showlegend=True
)

# Show the close-up comparison plot below the main plot
st.plotly_chart(fig_comparison)

# Add some explanatory text if needed
st.write("""
    This chart shows the comparison between the **actual temperatures** for 2022, 2023, and 2024 
    and the **forecasted temperatures** using the SARIMA model. The close-up chart below gives a more focused view 
    of how the predicted temperatures align with the actual ones throughout the year.
""")

# Calculate accuracy metrics
mae = mean_absolute_error(actual_all, forecast)  # Compare actual and forecasted for 36 months
rmse = np.sqrt(mean_squared_error(actual_all, forecast))

st.write(f"Mean Absolute Error (MAE): {mae:.2f}°F")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}°F")

st.write("""
       The Mean Absolute Error (MAE) of 2.19°F and Root Mean Squared Error (RMSE) of 2.67°F 
       indicate that the SARIMA model is doing a good job in forecasting the temperature for 
       the next 36 months. These are relatively low error values, suggesting that the forecasted 
       temperatures are close to the actual ones.
       """)
# Add this data to the historical data
actual_dates = pd.date_range('2022-01-01', periods=36, freq='M')
df_actual_2022_2024 = pd.DataFrame({
    'time': actual_dates,
    'Ftemp': actual_all
})

# Combine with existing data (assuming df already contains data until 2021)
df_combined = pd.concat([df, df_actual_2022_2024], ignore_index=True)

# Now, let's proceed with forecasting for 2025
# Convert the time column to datetime if it's not already in datetime format
df_combined['time'] = pd.to_datetime(df_combined['time'])

# Extract Year-Month from the time column
df_combined['YearMonth'] = df_combined['time'].dt.to_period('M')

# Calculate the monthly average temperature
monthly_avg = df_combined.groupby('YearMonth')['Ftemp'].mean()

# SARIMA Model
st.subheader("Fitting SARIMA Model")

# Define the SARIMA model (p, d, q) and (P, D, Q, S)
sarima_model = SARIMAX(monthly_avg, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
sarima_model_fit = sarima_model.fit(disp=False)

# Forecast the next 12 months (for 2025)
forecast_steps = 12
forecast_2025 = sarima_model_fit.forecast(steps=forecast_steps)

# Create the forecasted date range
forecast_index = pd.date_range(start=monthly_avg.index[-1].end_time + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

# Add a close-up chart comparing forecasted 2025 temperatures and actual data for 2022-2024
fig_comparison = go.Figure()

# Historical temperatures for 2022-2024 (red line)
fig_comparison.add_trace(go.Scatter(
    x=df_actual_2022_2024['time'], y=df_actual_2022_2024['Ftemp'],
    mode='lines+markers', name='Actual Temperature', 
    line=dict(color='red'), marker=dict(symbol='circle', size=8)
))

# Forecasted temperatures for 2025 (blue dashed line)
fig_comparison.add_trace(go.Scatter(
    x=forecast_index, y=forecast_2025,
    mode='lines+markers', name='Forecasted Temperature', 
    line=dict(color='blue', dash='dash'), marker=dict(symbol='circle', size=8)
))

# Update layout for the close-up comparison chart
fig_comparison.update_layout(
    title="Close-up: Predicted vs Actual Monthly Temperatures for 2022-2025",
    xaxis_title="Month",
    yaxis_title="Temperature (°F)",
    width=800,
    height=400,
    showlegend=True
)

# Show the close-up comparison plot below the main plot
st.plotly_chart(fig_comparison)

# Add some explanatory text if needed
st.write("""
    This chart shows the comparison between the **actual temperatures** for 2022-2024 
    and the **forecasted temperatures** for 2025 using the SARIMA model. The close-up chart below 
    gives a more focused view of how the predicted temperatures align with the actual ones over the months.
""")

# Generate the table to display the forecasted temperatures for 2025
forecast_table = pd.DataFrame({
    'Forecasted Temp (°F)': forecast_2025
})

# Display the table
st.subheader("2025 Monthly Forecasted Temperatures")
st.write(forecast_table)
