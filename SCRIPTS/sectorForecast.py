## Input: job_postings_by_sector_US.csv
## Ouput: top_sector_forecasts.png
## Function: Reads in data and then performs a linear regression to forecast predicitons for the top 10 hiring sectors in the next 10+ years on Indeed.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    if "indeed_job_postings_index_NSA" in df.columns:
        df.rename(columns={"indeed_job_postings_index_NSA": "job_postings"}, inplace=True)
    elif "indeed_job_postings_index" in df.columns:
        df.rename(columns={"indeed_job_postings_index": "job_postings"}, inplace=True)
    
    if 'sector' not in df.columns and 'display_name' in df.columns:
        df.rename(columns={'display_name': 'sector'}, inplace=True)
    
    if 'sector' not in df.columns:
        df = pd.melt(df, id_vars=["date"], var_name="sector", value_name="job_postings")
    
    return df

def sarimax_forecast_by_sector(df: pd.DataFrame, forecast_months=60):
    """
    For each sector:
      1. Sort & resample monthly (end of month).
      2. Fit a SARIMAX model (default: (1,1,1)x(1,0,1,12)).
      3. Compute a pseudo-R² from in-sample fits.
      4. Forecast the next 'forecast_months' steps.
    Returns a list of dicts: {
      'sector': ...,
      'r_squared': ...,
      'historical': (pd.Series),
      'forecast': (pd.Series),
      'forecast_final': (float)
    }
    """
    forecast_results = []
    sectors = df['sector'].unique()
    
    for sector in sectors:
        sector_df = df[df['sector'] == sector].copy()
        sector_df.sort_values('date', inplace=True)
        sector_df.set_index('date', inplace=True)
        
        monthly_series = sector_df['job_postings'].resample('ME').mean().dropna()
        
        if len(monthly_series) < 12:
            continue
        monthly_series = monthly_series.asfreq('ME')
        model = sm.tsa.statespace.SARIMAX(
            monthly_series,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        fitted_vals = results.fittedvalues
        fitted_vals = fitted_vals.reindex(monthly_series.index)
        sse = np.sum((monthly_series - fitted_vals)**2)
        sst = np.sum((monthly_series - monthly_series.mean())**2)
        r_squared = 1 - sse/sst if sst != 0 else 0
        
 
        forecast_object = results.get_forecast(steps=forecast_months)
        mean_forecast = forecast_object.predicted_mean

        forecast_results.append({
            'sector': sector,
            'r_squared': r_squared,
            'historical': monthly_series,
            'forecast': mean_forecast,
            'forecast_final': mean_forecast.iloc[-1]
        })
    
    return forecast_results

def plot_top_sector_forecasts(forecast_results, top_n=10, output_path="OUTPUTS/top_sector_forecasts.png"):
    """
    1. Sort the sectors by final forecast value (descending).
    2. Plot each sector's historical data (solid) + forecast (dashed).
    3. Draw a short bridging line (dotted) from last hist date to first forecast date 
       to ensure no visible gap.
    4. Legend includes sector name & R².
    """
    forecast_results.sort(key=lambda x: x['forecast_final'], reverse=True)
    top_results = forecast_results[:top_n]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, res in enumerate(top_results):
        color = colors[i % len(colors)]
        hist_index = res['historical'].index
        hist_values = res['historical'].values
        plt.plot(hist_index, hist_values, color=color)
        
        fc_index = res['forecast'].index
        fc_values = res['forecast'].values
        plt.plot(fc_index, fc_values, linestyle='--', color=color,
                 label=f"{res['sector']} (R²={res['r_squared']:.2f})")
        
        ## bridges the lines and adds a dash to connect possible gaps
        last_hist_date = hist_index[-1]
        last_hist_val = hist_values[-1]
        first_fc_date = fc_index[0]
        first_fc_val = fc_values[0]
        if first_fc_date > last_hist_date:
            plt.plot([last_hist_date, first_fc_date], [last_hist_val, first_fc_val],
                     color=color, linestyle=':')
    
    plt.title("Top 10 Sectors Forecast (SARIMAX)")
    plt.xlabel("Date")
    plt.ylabel("Job Postings")
    plt.legend(title="Sector")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_forecast():
    csv_path = "DATA/job_postings_by_sector_US.csv"
    df = load_data(csv_path)
    
    forecast_results = sarimax_forecast_by_sector(df, forecast_months=60)
    
    if forecast_results:
        plot_top_sector_forecasts(forecast_results, top_n=10)
        print("Forecast plot saved to OUTPUTS/top_sector_forecasts.png")
    else:
        print("No sectors had sufficient data for forecasting.")

if __name__ == "__main__":
    run_forecast()
