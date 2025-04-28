## Input: aggregate_job_postings.csv
## Ouput: job_postings_time_series_forecast.png
## Function: Reads in data and then performs a linear regression to forecast predicitons for the amount of Indeed job posts each day for the next 10+ years.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm  

def load_aggregate_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the CSV file, converts the 'date' column to datetime,
    and renames the job_postings column if needed.
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    if "indeed_job_postings_index_NSA" in df.columns:
        df.rename(columns={"indeed_job_postings_index_NSA": "job_postings"}, inplace=True)
    elif "indeed_job_postings_index" in df.columns:
        df.rename(columns={"indeed_job_postings_index": "job_postings"}, inplace=True)
    
    return df

def sarimax_forecast(df: pd.DataFrame, forecast_months=60):
    """
    1. Sorts the DataFrame by date and sets date as index.
    2. Resamples to monthly frequency and fits a SARIMAX model.
    3. Computes an in-sample R² for the fitted model.
    4. Produces a forecast (with confidence intervals) for 'forecast_months'.
    5. Returns the monthly historical DataFrame, forecast DataFrame, and the R².
    """
    df = df.sort_values('date').copy()
    df.set_index('date', inplace=True)
    monthly_df = df['job_postings'].resample('M').mean().to_frame()
    monthly_df.dropna(inplace=True)

    # Fit SARIMAX
    model = sm.tsa.statespace.SARIMAX(
        monthly_df['job_postings'],
        order=(1, 1, 1),             # (p,d,q) 
        seasonal_order=(1, 0, 1, 12),# (P,D,Q,m) with m=12 for monthly seasonality
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    # Compute in-sample fitted values & align them to the actual data
    fitted_vals = results.fittedvalues.reindex(monthly_df.index)
    # Calculate R² = 1 - SSE/SST
    sse = ((monthly_df['job_postings'] - fitted_vals) ** 2).sum()
    sst = ((monthly_df['job_postings'] - monthly_df['job_postings'].mean()) ** 2).sum()
    r_squared = 1 - sse / sst if sst != 0 else 0

    # Forecast
    forecast_object = results.get_forecast(steps=forecast_months)
    mean_forecast = forecast_object.predicted_mean
    conf_int = forecast_object.conf_int()

    forecast_df = pd.DataFrame({
        'forecast': mean_forecast,
        'lower_ci': conf_int.iloc[:, 0],
        'upper_ci': conf_int.iloc[:, 1]
    })
    
    return monthly_df, forecast_df, r_squared

def plot_sarimax_forecast(
    monthly_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    r_squared: float,
    output_path="OUTPUTS/job_postings_time_series_forecast.png"
):
    """
    Plots the historical monthly job postings (blue), the forecast (orange, dashed),
    and the confidence intervals (shaded). Displays the R² in the plot title.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    # Historical
    plt.plot(monthly_df.index, monthly_df['job_postings'], label='Historical', color='blue')
    # Forecast
    plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='orange', linestyle='--')
    # Confidence intervals
    plt.fill_between(
        forecast_df.index,
        forecast_df['lower_ci'],
        forecast_df['upper_ci'],
        color='orange', alpha=0.2,
        label='Confidence Interval'
    )
    
    # Optional short bridge line from the last historical point to the first forecast point
    last_hist_date = monthly_df.index[-1]
    last_hist_value = monthly_df['job_postings'].iloc[-1]
    first_fc_date = forecast_df.index[0]
    first_fc_value = forecast_df['forecast'].iloc[0]
    plt.plot([last_hist_date, first_fc_date],
             [last_hist_value, first_fc_value],
             color='blue', linestyle=':')

    plt.title(f"Monthly Job Postings Forecast (SARIMAX)\nR² = {r_squared:.3f}")
    plt.xlabel('Date')
    plt.ylabel('Job Postings (Monthly Mean)')
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

def run_time_series_forecast():
    """
    Main driver function:
      1. Loads the data.
      2. Runs SARIMAX forecasting.
      3. Plots the results with an R² displayed.
    """
    csv_path = "DATA/aggregate_job_postings_US.csv"
    df = load_aggregate_data(csv_path)
    
    monthly_df, forecast_df, r_squared = sarimax_forecast(df, forecast_months=60)
    
    output_plot = "OUTPUTS/job_postings_time_series_forecast.png"
    plot_sarimax_forecast(monthly_df, forecast_df, r_squared, output_path=output_plot)
    print(f"Forecast plot saved to {output_plot}")

if __name__ == "__main__":
    run_time_series_forecast()
